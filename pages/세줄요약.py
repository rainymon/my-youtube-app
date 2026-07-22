import re
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)


# ---------------------------------------------------------
# 1. Streamlit 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="유튜브 댓글 AI 분석",
    page_icon="💬",
    layout="wide",
)


# ---------------------------------------------------------
# 2. 앱에서 사용할 기본값
# ---------------------------------------------------------
EXAMPLE_URL_1 = "https://youtu.be/d95J8yzvjbQ?si=LfL5DLwCL8Pk077r"
EXAMPLE_URL_2 = "https://youtu.be/I9vK5EVTt0U?si=NEZ8L7MRuNvrzINa"

YOUTUBE_API_URL = (
    "https://www.googleapis.com/youtube/v3/commentThreads"
)

SOLAR_BASE_URL = "https://api.upstage.ai/v1"

# 모델 이름은 요청한 값을 그대로 사용합니다.
SOLAR_MODEL = "solar-open2"


# ---------------------------------------------------------
# 3. 세션 상태 초기화
#
# Streamlit은 버튼을 누를 때마다 코드를 다시 실행합니다.
# 수집한 댓글과 AI 요약을 세션에 저장하면 다시 실행되어도
# 현재 브라우저에서는 결과가 유지됩니다.
# ---------------------------------------------------------
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = EXAMPLE_URL_1

if "comments_df" not in st.session_state:
    st.session_state.comments_df = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None


# ---------------------------------------------------------
# 4. 예시 버튼에서 사용할 함수
#
# 예시 버튼을 누르면 입력창의 주소를 바꾸고,
# 이전에 가져온 댓글과 요약 결과는 지웁니다.
# ---------------------------------------------------------
def select_example(url: str):
    """선택한 예시 주소를 입력창에 넣습니다."""

    st.session_state.youtube_url = url
    st.session_state.comments_df = None
    st.session_state.video_id = None
    st.session_state.ai_summary = None


# ---------------------------------------------------------
# 5. 유튜브 주소에서 영상 ID 추출
#
# 지원 주소:
# - https://youtu.be/영상ID
# - https://www.youtube.com/watch?v=영상ID
# - https://youtube.com/shorts/영상ID
# - https://youtube.com/embed/영상ID
#
# ?si=... 등의 추가 주소 정보는 영상 ID에서 제외됩니다.
# ---------------------------------------------------------
def extract_video_id(url: str):
    """유튜브 주소에서 11자리 영상 ID를 추출합니다."""

    if not url:
        return None

    url = url.strip()

    # https://를 생략한 주소도 처리합니다.
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed_url = urlparse(url)
        hostname = (
            parsed_url.hostname.lower()
            if parsed_url.hostname
            else ""
        )
    except ValueError:
        return None

    video_id = None

    # youtu.be 형식의 짧은 주소
    if hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed_url.path.strip("/").split("/")[0]

    # youtube.com 형식의 주소
    elif hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    ):
        path = parsed_url.path.rstrip("/")

        # 일반 영상 주소
        # 예: youtube.com/watch?v=영상ID
        if path == "/watch":
            query_values = parse_qs(parsed_url.query)
            video_id = query_values.get("v", [None])[0]

        # Shorts 주소
        elif path.startswith("/shorts/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 임베드 주소
        elif path.startswith("/embed/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 이전 형식의 주소
        elif path.startswith("/v/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

    # 유튜브 영상 ID는 보통 영문, 숫자, -, _로 구성된 11자리입니다.
    if video_id and re.fullmatch(
        r"[A-Za-z0-9_-]{11}",
        video_id,
    ):
        return video_id

    return None


# ---------------------------------------------------------
# 6. YouTube API 오류 내용 추출
# ---------------------------------------------------------
def get_youtube_error_info(response):
    """YouTube API 오류 응답에서 사유와 메시지를 꺼냅니다."""

    try:
        error_data = response.json()
    except ValueError:
        return "", ""

    error_object = error_data.get("error", {})
    error_message = error_object.get("message", "")
    error_list = error_object.get("errors", [])

    if error_list:
        error_reason = error_list[0].get("reason", "")
    else:
        error_reason = ""

    return error_reason, error_message


# ---------------------------------------------------------
# 7. YouTube 댓글 가져오기
#
# YouTube API에는 relevance 순으로 요청합니다.
# 받아온 결과는 다시 좋아요 수 기준으로 정렬합니다.
# ---------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_youtube_comments(video_id: str, api_key: str):
    """YouTube 최상위 댓글을 최대 100개 가져옵니다."""

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "order": "relevance",
        "textFormat": "plainText",
        "key": api_key,
    }

    try:
        response = requests.get(
            YOUTUBE_API_URL,
            params=params,
            timeout=20,
        )

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "YouTube 서버의 응답이 늦어지고 있습니다. "
            "잠시 후 다시 시도해 주세요."
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "YouTube 서버에 연결하지 못했습니다. "
            "인터넷 연결 상태를 확인해 주세요."
        ) from error

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            "YouTube API를 요청하는 중 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        ) from error

    # 정상 응답이 아니라면 YouTube가 보낸 오류를 확인합니다.
    if not response.ok:
        error_reason, error_message = get_youtube_error_info(
            response
        )

        if error_reason == "commentsDisabled":
            raise RuntimeError(
                "이 영상은 댓글 사용이 중지되어 있어 "
                "댓글을 가져올 수 없습니다."
            )

        if error_reason in ("videoNotFound", "notFound"):
            raise RuntimeError(
                "영상을 찾을 수 없습니다. "
                "삭제되었거나 비공개 영상인지 확인해 주세요."
            )

        if error_reason in (
            "keyInvalid",
            "badRequest",
            "accessNotConfigured",
        ):
            raise RuntimeError(
                "YouTube API 키가 올바르지 않거나 "
                "YouTube Data API v3가 활성화되지 않았습니다."
            )

        if error_reason in (
            "quotaExceeded",
            "dailyLimitExceeded",
            "dailyLimitExceededUnreg",
        ):
            raise RuntimeError(
                "오늘 사용할 수 있는 YouTube API 할당량을 "
                "모두 사용했습니다."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "댓글에 접근할 수 없습니다. "
                "댓글 공개 여부와 API 설정을 확인해 주세요."
            )

        if response.status_code == 404:
            raise RuntimeError(
                "영상을 찾을 수 없습니다. "
                "유튜브 링크를 다시 확인해 주세요."
            )

        detail = (
            f" 상세 내용: {error_message}"
            if error_message
            else ""
        )

        raise RuntimeError(
            f"YouTube 댓글 요청에 실패했습니다. "
            f"오류 코드: {response.status_code}.{detail}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "YouTube 서버에서 올바르지 않은 형식의 "
            "응답을 받았습니다."
        ) from error

    items = result.get("items", [])
    comments = []

    # commentThreads 안에서 최상위 댓글 정보를 꺼냅니다.
    for item in items:
        try:
            comment_snippet = (
                item["snippet"]["topLevelComment"]["snippet"]
            )

            # 댓글 원문은 textOriginal 필드를 사용합니다.
            comment_text = comment_snippet.get(
                "textOriginal",
                "",
            )

            # 좋아요 수는 likeCount 필드를 사용합니다.
            like_count = pd.to_numeric(
                comment_snippet.get("likeCount", 0),
                errors="coerce",
            )

            if pd.isna(like_count):
                like_count = 0

            # 내용이 있는 댓글만 저장합니다.
            if str(comment_text).strip():
                comments.append(
                    {
                        "댓글": str(comment_text).strip(),
                        "좋아요 수": int(like_count),
                    }
                )

        # 구조가 불완전한 댓글은 그 댓글만 건너뜁니다.
        except (KeyError, TypeError):
            continue

    if not comments:
        return pd.DataFrame(
            columns=["순번", "댓글", "좋아요 수"]
        )

    comments_df = pd.DataFrame(comments)

    # 좋아요 수를 실제 숫자로 다시 확인합니다.
    comments_df["좋아요 수"] = pd.to_numeric(
        comments_df["좋아요 수"],
        errors="coerce",
    ).fillna(0).astype(int)

    # 좋아요가 많은 댓글부터 정렬합니다.
    comments_df = comments_df.sort_values(
        by="좋아요 수",
        ascending=False,
    ).reset_index(drop=True)

    # 표에 표시할 순번을 붙입니다.
    comments_df.insert(
        0,
        "순번",
        range(1, len(comments_df) + 1),
    )

    return comments_df


# ---------------------------------------------------------
# 8. Solar에 전달할 댓글 텍스트 만들기
#
# 댓글마다 번호와 좋아요 수를 함께 전달합니다.
# 이렇게 하면 AI가 반응의 중요도를 판단하는 데 도움이 됩니다.
# ---------------------------------------------------------
def make_comments_text(comments_df: pd.DataFrame):
    """데이터프레임의 댓글 전체를 하나의 텍스트로 만듭니다."""

    comment_lines = []

    for index, row in comments_df.iterrows():
        comment_lines.append(
            f"{index + 1}. "
            f"[좋아요 {int(row['좋아요 수']):,}개] "
            f"{row['댓글']}"
        )

    return "\n".join(comment_lines)


# ---------------------------------------------------------
# 9. Solar로 댓글 전체 반응 요약
# ---------------------------------------------------------
def summarize_comments_with_solar(
    comments_df: pd.DataFrame,
    api_key: str,
):
    """Solar 모델로 댓글 전체 반응을 한국어 세 줄로 요약합니다."""

    client = OpenAI(
        api_key=api_key,
        base_url=SOLAR_BASE_URL,
        timeout=60.0,
        max_retries=1,
    )

    comments_text = make_comments_text(comments_df)

    system_prompt = """
너는 유튜브 댓글 반응을 객관적으로 분석하는 전문가야.
반드시 한국어로만 답해.
사용자가 제공한 댓글에 없는 사실은 만들지 마.
긍정과 부정의 비율은 댓글의 표현을 바탕으로 대략 추정해.
출력은 반드시 정확히 세 줄로 작성해.
제목, 번호, 글머리표, 인사말, 추가 설명은 쓰지 마.
세 번째 줄 끝에는 반드시
'긍정 약 00%, 부정 약 00%' 형식의 추정치를 포함해.
긍정과 부정 비율의 합은 100%가 되게 해.
중립적인 반응은 더 가까운 쪽에 포함해 대략 추정해.
""".strip()

    user_prompt = f"""
다음은 유튜브 영상에서 가져온 댓글 전체야.

전체 댓글의 공통 반응과 핵심 의견을 한국어 세 줄로 요약해 줘.
좋아요 수가 많은 댓글은 많은 시청자가 공감한 반응일 수 있으므로
분석할 때 참고해.

댓글:
{comments_text}
""".strip()

    response = client.chat.completions.create(
        # 모델 이름은 반드시 solar-open2를 그대로 사용합니다.
        model=SOLAR_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        # 추론 기능을 끕니다.
        # temperature가 아니라 reasoning_effort를 사용합니다.
        reasoning_effort="none",
    )

    if not response.choices:
        raise RuntimeError(
            "Solar가 요약 결과를 보내지 않았습니다."
        )

    summary = response.choices[0].message.content

    if not summary or not summary.strip():
        raise RuntimeError(
            "Solar가 빈 요약 결과를 보냈습니다."
        )

    return summary.strip()


# ---------------------------------------------------------
# 10. 화면 제목
# ---------------------------------------------------------
st.title("💬 유튜브 댓글 AI 분석")
st.caption(
    "유튜브 인기 댓글을 최대 100개 가져오고 "
    "Solar 인공지능으로 전체 반응을 세 줄로 요약합니다."
)


# ---------------------------------------------------------
# 11. 예시 영상 선택 버튼
# ---------------------------------------------------------
st.subheader("예시 영상 선택")

example_column_1, example_column_2 = st.columns(2)

with example_column_1:
    st.button(
        "예시 1 · 딥마인드 다큐(영어 댓글)",
        use_container_width=True,
        on_click=select_example,
        args=(EXAMPLE_URL_1,),
    )

with example_column_2:
    st.button(
        "예시 2 · 2002 월드컵 추억(한국어 댓글)",
        use_container_width=True,
        on_click=select_example,
        args=(EXAMPLE_URL_2,),
    )


# ---------------------------------------------------------
# 12. 유튜브 주소 입력창
# ---------------------------------------------------------
video_url = st.text_input(
    "유튜브 영상 링크",
    key="youtube_url",
    placeholder="https://www.youtube.com/watch?v=...",
    help=(
        "youtu.be 짧은 주소와 youtube.com/watch 주소를 "
        "모두 사용할 수 있습니다."
    ),
)


# ---------------------------------------------------------
# 13. 댓글 가져오기 버튼
# ---------------------------------------------------------
fetch_button = st.button(
    "댓글 가져오기",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# 14. 댓글 가져오기 버튼을 눌렀을 때 실행
# ---------------------------------------------------------
if fetch_button:
    video_id = extract_video_id(video_url)

    if video_id is None:
        st.error(
            "올바른 유튜브 영상 링크를 입력해 주세요. "
            "youtu.be 또는 youtube.com/watch 형식을 사용할 수 있습니다."
        )

    else:
        # Secrets에서 YouTube API 키를 불러옵니다.
        try:
            youtube_api_key = str(
                st.secrets["YOUTUBE_API_KEY"]
            ).strip()

        except KeyError:
            st.error(
                "YouTube API 키가 등록되지 않았습니다. "
                "Streamlit Cloud의 비밀 금고에 "
                "`YOUTUBE_API_KEY`를 등록해 주세요."
            )

            st.code(
                'YOUTUBE_API_KEY = "발급받은_YouTube_API_키"',
                language="toml",
            )

            youtube_api_key = ""

        if youtube_api_key:
            try:
                with st.spinner(
                    "유튜브 댓글을 가져오는 중입니다..."
                ):
                    new_comments_df = fetch_youtube_comments(
                        video_id=video_id,
                        api_key=youtube_api_key,
                    )

                if new_comments_df.empty:
                    # 새 요청에 댓글이 없으면 이전 결과도 지웁니다.
                    st.session_state.comments_df = None
                    st.session_state.video_id = None
                    st.session_state.ai_summary = None

                    st.warning(
                        "가져올 수 있는 댓글이 없습니다. "
                        "댓글이 없거나 댓글 공개가 제한되었을 수 있습니다."
                    )

                else:
                    # 가져온 댓글과 영상 ID를 세션에 저장합니다.
                    st.session_state.comments_df = new_comments_df
                    st.session_state.video_id = video_id

                    # 새로운 영상을 불러왔으므로 이전 요약은 지웁니다.
                    st.session_state.ai_summary = None

                    st.success(
                        "댓글을 성공적으로 가져왔습니다."
                    )

            except RuntimeError as error:
                st.error(str(error))
                st.info(
                    "영상 링크, 댓글 공개 여부, API 키와 "
                    "API 할당량을 확인해 주세요."
                )

            except Exception:
                st.error(
                    "댓글을 처리하는 중 예상하지 못한 문제가 발생했습니다. "
                    "잠시 후 다시 시도해 주세요."
                )

        elif "YOUTUBE_API_KEY" in st.secrets:
            st.error(
                "등록된 YouTube API 키가 비어 있습니다. "
                "비밀 금고 설정을 확인해 주세요."
            )


# ---------------------------------------------------------
# 15. 세션에 댓글이 저장되어 있으면 계속 화면에 표시
# ---------------------------------------------------------
comments_df = st.session_state.comments_df

if comments_df is not None and not comments_df.empty:
    st.divider()

    if st.session_state.video_id:
        st.caption(
            f"추출된 영상 ID: `{st.session_state.video_id}`"
        )

    # 댓글 개수를 큰 지표 카드로 표시합니다.
    st.metric(
        label="가져온 댓글 수",
        value=f"{len(comments_df):,}개",
    )

    st.subheader("👍 좋아요가 많은 댓글")

    # 좋아요 수가 많은 순서로 정렬된 댓글 표입니다.
    st.dataframe(
        comments_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "순번": st.column_config.NumberColumn(
                "순번",
                format="%d",
                width="small",
            ),
            "댓글": st.column_config.TextColumn(
                "댓글 원문",
                width="large",
            ),
            "좋아요 수": st.column_config.NumberColumn(
                "좋아요 수",
                format="%,d",
                width="small",
            ),
        },
    )

    st.divider()

    # -----------------------------------------------------
    # 16. AI 세 줄 요약 버튼
    # -----------------------------------------------------
    st.subheader("🤖 댓글 전체 반응 요약")

    summary_button = st.button(
        "AI 세 줄 요약",
        type="secondary",
        use_container_width=True,
    )

    if summary_button:
        # Secrets에서 Solar API 키를 불러옵니다.
        try:
            solar_api_key = str(
                st.secrets["SOLAR_API_KEY"]
            ).strip()

        except KeyError:
            st.error(
                "Solar API 키가 등록되지 않았습니다. "
                "Streamlit Cloud의 비밀 금고에 "
                "`SOLAR_API_KEY`를 등록해 주세요."
            )

            st.code(
                'SOLAR_API_KEY = "발급받은_Solar_API_키"',
                language="toml",
            )

            solar_api_key = ""

        if solar_api_key:
            try:
                with st.spinner(
                    "Solar가 댓글 전체 반응을 분석하는 중입니다..."
                ):
                    summary = summarize_comments_with_solar(
                        comments_df=comments_df,
                        api_key=solar_api_key,
                    )

                # 완성된 요약을 세션에 저장합니다.
                st.session_state.ai_summary = summary

            except AuthenticationError:
                st.error(
                    "Solar API 인증에 실패했습니다. "
                    "비밀 금고에 등록한 `SOLAR_API_KEY`가 "
                    "올바른지 확인해 주세요."
                )

            except RateLimitError:
                st.error(
                    "Solar API 사용 한도 또는 요청 횟수 제한에 "
                    "도달했습니다. 잠시 후 다시 시도해 주세요."
                )

            except APIConnectionError:
                st.error(
                    "Solar API 서버에 연결하지 못했습니다. "
                    "인터넷 연결 상태를 확인한 뒤 다시 시도해 주세요."
                )

            except APIStatusError as error:
                if error.status_code == 400:
                    st.error(
                        "Solar가 요약 요청을 처리하지 못했습니다. "
                        "댓글의 양이 너무 많거나 요청 형식에 "
                        "문제가 있을 수 있습니다."
                    )

                elif error.status_code in (401, 403):
                    st.error(
                        "Solar API 키 또는 사용 권한을 확인해 주세요."
                    )

                elif error.status_code == 404:
                    st.error(
                        "Solar 모델 또는 API 주소를 찾지 못했습니다."
                    )

                elif error.status_code >= 500:
                    st.error(
                        "현재 Solar 서버에 일시적인 문제가 있습니다. "
                        "잠시 후 다시 시도해 주세요."
                    )

                else:
                    st.error(
                        "Solar 요약 요청 중 문제가 발생했습니다. "
                        f"오류 코드: {error.status_code}"
                    )

            except RuntimeError as error:
                st.error(str(error))

            except Exception:
                st.error(
                    "댓글을 요약하는 중 예상하지 못한 문제가 발생했습니다. "
                    "잠시 후 다시 시도해 주세요."
                )

        elif "SOLAR_API_KEY" in st.secrets:
            st.error(
                "등록된 Solar API 키가 비어 있습니다. "
                "비밀 금고 설정을 확인해 주세요."
            )

    # 저장된 AI 요약이 있으면 버튼을 누른 뒤에도 계속 표시합니다.
    if st.session_state.ai_summary:
        st.success("AI 요약이 완료되었습니다.")

        st.markdown(
            st.session_state.ai_summary
        )

else:
    st.info(
        "유튜브 영상 링크를 입력한 뒤 "
        "‘댓글 가져오기’ 버튼을 눌러 주세요."
    )
