import re
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


# =========================================================
# 1. Streamlit 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="유튜브 댓글 AI 분석",
    page_icon="💬",
    layout="wide",
)


# =========================================================
# 2. 앱에서 사용할 고정 설정
# =========================================================

# 예시 영상 주소
EXAMPLE_URL_1 = "https://youtu.be/d95J8yzvjbQ?si=LfL5DLwCL8Pk077r"
EXAMPLE_URL_2 = "https://youtu.be/I9vK5EVTt0U?si=NEZ8L7MRuNvrzINa"

# YouTube Data API 댓글 요청 주소
YOUTUBE_API_URL = (
    "https://www.googleapis.com/youtube/v3/commentThreads"
)

# Solar API 설정
SOLAR_BASE_URL = "https://api.upstage.ai/v1"

# 요청한 모델 이름을 글자 그대로 사용합니다.
SOLAR_MODEL = "solar-open2"


# =========================================================
# 3. 세션 상태 초기화
#
# Streamlit은 버튼을 누를 때마다 코드를 다시 실행합니다.
# 세션 상태에 저장하면 수집한 댓글과 AI 결과가 유지됩니다.
# =========================================================
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = EXAMPLE_URL_1

if "comments_df" not in st.session_state:
    st.session_state.comments_df = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None

if "comment_answer" not in st.session_state:
    st.session_state.comment_answer = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# =========================================================
# 4. 이전 분석 결과를 지우는 함수
# =========================================================
def clear_analysis_results():
    """저장된 댓글과 AI 분석 결과를 모두 지웁니다."""

    st.session_state.comments_df = None
    st.session_state.video_id = None
    st.session_state.ai_summary = None
    st.session_state.comment_answer = None
    st.session_state.last_question = ""


# =========================================================
# 5. 예시 영상 버튼에서 실행할 함수
# =========================================================
def select_example(url: str):
    """선택한 예시 링크를 입력창에 넣습니다."""

    st.session_state.youtube_url = url
    clear_analysis_results()


# =========================================================
# 6. 유튜브 링크에서 영상 ID 추출
#
# 지원하는 링크:
# - https://youtu.be/영상ID
# - https://www.youtube.com/watch?v=영상ID
# - https://youtube.com/shorts/영상ID
# - https://youtube.com/embed/영상ID
#
# ?si=... 같은 추가 정보는 자동으로 무시됩니다.
# =========================================================
def extract_video_id(url: str):
    """유튜브 링크에서 11자리 영상 ID를 추출합니다."""

    if not url:
        return None

    url = url.strip()

    # 사용자가 https://를 생략했을 때 자동으로 붙입니다.
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

    # youtu.be 형식의 짧은 링크
    if hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed_url.path.strip("/").split("/")[0]

    # youtube.com 형식의 링크
    elif hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    ):
        path = parsed_url.path.rstrip("/")

        # 일반 영상 링크
        # 예: youtube.com/watch?v=영상ID
        if path == "/watch":
            query_values = parse_qs(parsed_url.query)
            video_id = query_values.get("v", [None])[0]

        # Shorts 링크
        # 예: youtube.com/shorts/영상ID
        elif path.startswith("/shorts/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 영상 삽입 링크
        # 예: youtube.com/embed/영상ID
        elif path.startswith("/embed/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 이전 형식의 링크
        # 예: youtube.com/v/영상ID
        elif path.startswith("/v/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

    # 유튜브 영상 ID는 영문자, 숫자, -, _로 구성된 11자리입니다.
    if video_id and re.fullmatch(
        r"[A-Za-z0-9_-]{11}",
        video_id,
    ):
        return video_id

    return None


# =========================================================
# 7. YouTube API 오류 내용 추출
# =========================================================
def get_youtube_error_info(response: requests.Response):
    """YouTube 오류 응답에서 오류 사유와 메시지를 꺼냅니다."""

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


# =========================================================
# 8. YouTube 댓글 가져오기
#
# API에는 관련도순(order=relevance)으로 최대 100개를 요청하고,
# 받은 뒤 좋아요 수 기준으로 다시 내림차순 정렬합니다.
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_youtube_comments(
    video_id: str,
    api_key: str,
) -> pd.DataFrame:
    """YouTube의 최상위 댓글을 최대 100개 가져옵니다."""

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

    # HTTP 요청이 실패했다면 YouTube가 보낸 오류를 확인합니다.
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
                "영상이 삭제되었거나 비공개인지 확인해 주세요."
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

    # commentThreads 목록에서 최상위 댓글 정보를 꺼냅니다.
    for item in items:
        try:
            comment_snippet = (
                item["snippet"]["topLevelComment"]["snippet"]
            )

            # 댓글 원문은 textOriginal을 사용합니다.
            comment_text = str(
                comment_snippet.get("textOriginal", "")
            ).strip()

            # 좋아요 수는 likeCount를 사용합니다.
            like_count = pd.to_numeric(
                comment_snippet.get("likeCount", 0),
                errors="coerce",
            )

            if pd.isna(like_count):
                like_count = 0

            # 내용이 있는 댓글만 저장합니다.
            if comment_text:
                comments.append(
                    {
                        "댓글": comment_text,
                        "좋아요 수": int(like_count),
                    }
                )

        # 구조가 불완전한 댓글은 해당 댓글만 건너뜁니다.
        except (KeyError, TypeError):
            continue

    if not comments:
        return pd.DataFrame(
            columns=["순번", "댓글", "좋아요 수"]
        )

    comments_df = pd.DataFrame(comments)

    # 좋아요 수를 실제 숫자로 변환합니다.
    comments_df["좋아요 수"] = pd.to_numeric(
        comments_df["좋아요 수"],
        errors="coerce",
    ).fillna(0).astype(int)

    # 좋아요 수가 많은 댓글부터 정렬합니다.
    comments_df = comments_df.sort_values(
        by="좋아요 수",
        ascending=False,
    ).reset_index(drop=True)

    # 표에 표시할 순번을 추가합니다.
    comments_df.insert(
        0,
        "순번",
        range(1, len(comments_df) + 1),
    )

    return comments_df


# =========================================================
# 9. 댓글 전체를 Solar에 보낼 문자열로 변환
# =========================================================
def make_comments_text(comments_df: pd.DataFrame) -> str:
    """댓글과 좋아요 수를 하나의 긴 문자열로 만듭니다."""

    comment_lines = []

    for index, row in comments_df.iterrows():
        comment_lines.append(
            f"{index + 1}. "
            f"[좋아요 {int(row['좋아요 수']):,}개] "
            f"{row['댓글']}"
        )

    return "\n".join(comment_lines)


# =========================================================
# 10. Solar API 클라이언트 만들기
# =========================================================
def create_solar_client(api_key: str) -> OpenAI:
    """OpenAI 라이브러리로 Solar API에 연결합니다."""

    return OpenAI(
        api_key=api_key,
        base_url=SOLAR_BASE_URL,
        timeout=60.0,
        max_retries=1,
    )


# =========================================================
# 11. Solar로 댓글 전체를 세 줄 요약
# =========================================================
def summarize_comments_with_solar(
    comments_df: pd.DataFrame,
    api_key: str,
) -> str:
    """댓글 전체 반응을 한국어 세 줄로 요약합니다."""

    client = create_solar_client(api_key)
    comments_text = make_comments_text(comments_df)

    system_prompt = """
너는 유튜브 댓글의 전체 반응을 객관적으로 분석하는 전문가야.
반드시 한국어로만 답해.
제공된 댓글에 없는 사실은 추측하거나 만들어 내지 마.
댓글에서 공통으로 나타난 반응과 핵심 의견을 중심으로 분석해.
좋아요 수가 많은 댓글은 많은 사람이 공감했을 가능성이 있으므로 참고해.
다만 일부 댓글만으로 모든 시청자의 생각이라고 단정하지 마.

출력은 반드시 정확히 세 줄로 작성해.
제목, 번호, 글머리표, 인사말, 추가 설명은 쓰지 마.
첫 번째 줄에는 전체적인 반응을 요약해.
두 번째 줄에는 많이 나타난 핵심 의견이나 쟁점을 요약해.
세 번째 줄에는 감정 반응을 요약하고,
문장 끝에 반드시 '긍정 약 00%, 부정 약 00%' 형식으로 적어.
긍정과 부정 비율의 합은 반드시 100%가 되게 해.
중립적인 반응은 문맥상 더 가까운 쪽에 포함해 대략 추정해.
""".strip()

    user_prompt = f"""
다음은 한 유튜브 영상에서 가져온 댓글 전체야.

댓글 전체의 반응을 한국어 세 줄로 요약해 줘.

[댓글 자료]
{comments_text}
""".strip()

    response = client.chat.completions.create(
        # 모델 이름은 요청한 이름을 그대로 사용합니다.
        model="solar-open2",
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


# =========================================================
# 12. 댓글 전체를 근거로 자유 질문
# =========================================================
def ask_solar_about_comments(
    comments_df: pd.DataFrame,
    question: str,
    api_key: str,
) -> str:
    """가져온 댓글만을 근거로 사용자의 질문에 답합니다."""

    client = create_solar_client(api_key)
    comments_text = make_comments_text(comments_df)

    system_prompt = """
너는 유튜브 댓글을 근거로 질문에 답하는 분석가야.
반드시 한국어로만 답해.

제공된 댓글 자료만을 근거로 답해.
댓글에 없는 사실은 절대로 추측하거나 지어내지 마.
댓글만으로 답할 수 없다면
'제공된 댓글만으로는 확인하기 어렵습니다'라고 분명히 말해.

여러 댓글에서 반복되는 의견을 우선해서 설명해.
좋아요가 많은 댓글은 많은 사람이 공감했을 가능성이 있으므로 참고해.
하지만 수집된 댓글 일부를 모든 시청자의 의견이라고 단정하지 마.
가능하면 어떤 댓글 반응을 근거로 판단했는지 함께 설명해.
답변은 이해하기 쉬운 한국어로 작성해.
""".strip()

    user_prompt = f"""
다음 댓글 자료만을 근거로 사용자의 질문에 답해 줘.

[댓글 자료]
{comments_text}

[사용자 질문]
{question}
""".strip()

    response = client.chat.completions.create(
        # 모델 이름은 반드시 solar-open2를 사용합니다.
        model="solar-open2",
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
        reasoning_effort="none",
    )

    if not response.choices:
        raise RuntimeError(
            "Solar가 질문에 대한 답변을 보내지 않았습니다."
        )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError(
            "Solar가 빈 답변을 보냈습니다."
        )

    return answer.strip()


# =========================================================
# 13. Solar API 키를 비밀 금고에서 읽는 함수
# =========================================================
def get_solar_api_key():
    """Streamlit Secrets에서 Solar API 키를 불러옵니다."""

    try:
        api_key = str(
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

        return None

    if not api_key:
        st.error(
            "등록된 Solar API 키가 비어 있습니다. "
            "비밀 금고 설정을 확인해 주세요."
        )
        return None

    return api_key


# =========================================================
# 14. Solar API 오류를 친절하게 표시하는 함수
# =========================================================
def show_solar_error(error):
    """Solar 요청 실패 원인에 맞는 한국어 메시지를 표시합니다."""

    if isinstance(error, AuthenticationError):
        st.error(
            "Solar API 인증에 실패했습니다. "
            "비밀 금고의 `SOLAR_API_KEY`가 올바른지 확인해 주세요."
        )

    elif isinstance(error, RateLimitError):
        st.error(
            "Solar API 사용 한도 또는 요청 횟수 제한에 도달했습니다. "
            "잠시 후 다시 시도해 주세요."
        )

    elif isinstance(error, APIConnectionError):
        st.error(
            "Solar API 서버에 연결하지 못했습니다. "
            "인터넷 연결 상태를 확인한 뒤 다시 시도해 주세요."
        )

    elif isinstance(error, APIStatusError):
        if error.status_code == 400:
            st.error(
                "Solar가 요청 내용을 처리하지 못했습니다. "
                "댓글의 양이나 질문 내용을 확인해 주세요."
            )

        elif error.status_code in (401, 403):
            st.error(
                "Solar API 키가 올바르지 않거나 "
                "API를 사용할 권한이 없습니다."
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
                "Solar 요청을 처리하는 중 문제가 발생했습니다. "
                f"오류 코드: {error.status_code}"
            )

    elif isinstance(error, RuntimeError):
        st.error(str(error))

    else:
        st.error(
            "Solar의 답변을 받는 중 예상하지 못한 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )


# =========================================================
# 15. 앱 화면 제목
# =========================================================
st.title("💬 유튜브 댓글 AI 분석")
st.caption(
    "인기 댓글을 최대 100개 가져오고, "
    "Solar 인공지능으로 댓글 반응을 분석합니다."
)


# =========================================================
# 16. 예시 영상 선택 버튼
# =========================================================
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


# =========================================================
# 17. 유튜브 링크 입력창
# =========================================================
video_url = st.text_input(
    "유튜브 영상 링크",
    key="youtube_url",
    placeholder="https://www.youtube.com/watch?v=...",
    help=(
        "youtu.be 짧은 링크와 youtube.com/watch 링크를 "
        "모두 사용할 수 있습니다."
    ),
)


# =========================================================
# 18. 댓글 가져오기 버튼
# =========================================================
fetch_button = st.button(
    "댓글 가져오기",
    type="primary",
    use_container_width=True,
)


# =========================================================
# 19. 댓글 가져오기 실행
# =========================================================
if fetch_button:
    video_id = extract_video_id(video_url)

    if video_id is None:
        clear_analysis_results()

        st.error(
            "올바른 유튜브 영상 링크를 입력해 주세요. "
            "youtu.be 또는 youtube.com/watch 형식을 사용할 수 있습니다."
        )

    else:
        # YouTube API 키를 비밀 금고에서 가져옵니다.
        try:
            youtube_api_key = str(
                st.secrets["YOUTUBE_API_KEY"]
            ).strip()

        except KeyError:
            youtube_api_key = ""

            st.error(
                "YouTube API 키가 등록되지 않았습니다. "
                "Streamlit Cloud의 비밀 금고에 "
                "`YOUTUBE_API_KEY`를 등록해 주세요."
            )

            st.code(
                'YOUTUBE_API_KEY = "발급받은_YouTube_API_키"',
                language="toml",
            )

        if not youtube_api_key:
            if "YOUTUBE_API_KEY" in st.secrets:
                st.error(
                    "등록된 YouTube API 키가 비어 있습니다. "
                    "비밀 금고 설정을 확인해 주세요."
                )

        else:
            try:
                with st.spinner(
                    "유튜브 댓글을 가져오는 중입니다..."
                ):
                    new_comments_df = fetch_youtube_comments(
                        video_id=video_id,
                        api_key=youtube_api_key,
                    )

                if new_comments_df.empty:
                    clear_analysis_results()

                    st.warning(
                        "가져올 수 있는 댓글이 없습니다. "
                        "댓글이 없거나 댓글 공개가 제한되었을 수 있습니다."
                    )

                else:
                    # 댓글과 영상 ID를 세션에 저장합니다.
                    st.session_state.comments_df = new_comments_df
                    st.session_state.video_id = video_id

                    # 새 영상을 불러왔으므로 이전 AI 결과를 지웁니다.
                    st.session_state.ai_summary = None
                    st.session_state.comment_answer = None
                    st.session_state.last_question = ""

                    st.success(
                        "댓글을 성공적으로 가져왔습니다."
                    )

            except RuntimeError as error:
                clear_analysis_results()

                st.error(str(error))
                st.info(
                    "영상 링크, 댓글 공개 여부, API 키와 "
                    "API 할당량을 확인해 주세요."
                )

            except Exception:
                clear_analysis_results()

                st.error(
                    "댓글을 처리하는 중 예상하지 못한 문제가 발생했습니다. "
                    "잠시 후 다시 시도해 주세요."
                )


# =========================================================
# 20. 저장된 댓글 표시
# =========================================================
comments_df = st.session_state.comments_df

if comments_df is not None and not comments_df.empty:
    st.divider()

    if st.session_state.video_id:
        st.caption(
            f"추출된 영상 ID: `{st.session_state.video_id}`"
        )

    # 가져온 댓글 개수를 크게 표시합니다.
    st.metric(
        label="가져온 댓글 수",
        value=f"{len(comments_df):,}개",
    )

    st.subheader("👍 좋아요가 많은 댓글")

    # 좋아요 수 기준으로 정렬된 댓글을 표로 보여줍니다.
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

    # =====================================================
    # 21. AI 세 줄 요약
    # =====================================================
    st.subheader("🤖 댓글 전체 반응 세 줄 요약")

    summary_button = st.button(
        "AI 세 줄 요약",
        use_container_width=True,
    )

    if summary_button:
        solar_api_key = get_solar_api_key()

        if solar_api_key:
            try:
                with st.spinner(
                    "Solar가 댓글 전체 반응을 분석하는 중입니다..."
                ):
                    st.session_state.ai_summary = (
                        summarize_comments_with_solar(
                            comments_df=comments_df,
                            api_key=solar_api_key,
                        )
                    )

            except (
                AuthenticationError,
                RateLimitError,
                APIConnectionError,
                APIStatusError,
                RuntimeError,
            ) as error:
                show_solar_error(error)

            except Exception as error:
                show_solar_error(error)

    # 저장된 요약은 화면이 다시 실행되어도 계속 표시됩니다.
    if st.session_state.ai_summary:
        st.success("AI 요약이 완료되었습니다.")
        st.markdown(st.session_state.ai_summary)

    st.divider()

    # =====================================================
    # 22. 댓글 근거 자유 질문
    # =====================================================
    st.subheader("💡 댓글에 관해 질문하기")

    st.caption(
        "가져온 댓글 전체를 근거로 Solar가 답합니다. "
        "댓글에서 확인할 수 없는 사실은 지어내지 않도록 설정했습니다."
    )

    comment_question = st.text_input(
        "질문",
        placeholder=(
            "예: 이 영상 반응이 어때? "
            "또는 가장 많이 나온 불만은?"
        ),
        key="question_input",
    )

    ask_button = st.button(
        "질문하기",
        type="primary",
        use_container_width=True,
    )

    if ask_button:
        cleaned_question = comment_question.strip()

        if not cleaned_question:
            st.warning(
                "댓글에 관해 궁금한 내용을 입력해 주세요."
            )

        else:
            solar_api_key = get_solar_api_key()

            if solar_api_key:
                try:
                    with st.spinner(
                        "Solar가 댓글을 살펴보고 답변하는 중입니다..."
                    ):
                        answer = ask_solar_about_comments(
                            comments_df=comments_df,
                            question=cleaned_question,
                            api_key=solar_api_key,
                        )

                    # 질문과 답변을 세션에 저장합니다.
                    st.session_state.last_question = cleaned_question
                    st.session_state.comment_answer = answer

                except (
                    AuthenticationError,
                    RateLimitError,
                    APIConnectionError,
                    APIStatusError,
                    RuntimeError,
                ) as error:
                    show_solar_error(error)

                except Exception as error:
                    show_solar_error(error)

    # 저장된 질문과 답변을 계속 표시합니다.
    if st.session_state.comment_answer:
        st.markdown("#### 질문")

        st.info(
            st.session_state.last_question
        )

        st.markdown("#### Solar의 답변")

        st.success(
            st.session_state.comment_answer
        )

else:
    st.info(
        "유튜브 영상 링크를 입력한 뒤 "
        "‘댓글 가져오기’ 버튼을 눌러 주세요."
    )
