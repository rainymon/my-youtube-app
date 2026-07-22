
import re
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# 1. Streamlit 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="유튜브 댓글 분석",
    page_icon="💬",
    layout="wide",
)


# ---------------------------------------------------------
# 2. 예시 영상 주소
# ---------------------------------------------------------
EXAMPLE_URL_1 = "https://youtu.be/d95J8yzvjbQ?si=LfL5DLwCL8Pk077r"
EXAMPLE_URL_2 = "https://youtu.be/I9vK5EVTt0U?si=NEZ8L7MRuNvrzINa"


# ---------------------------------------------------------
# 3. 유튜브 링크에서 영상 ID를 추출하는 함수
#
# 처리할 수 있는 주소 예시:
# - https://youtu.be/영상ID
# - https://www.youtube.com/watch?v=영상ID
# - https://youtube.com/shorts/영상ID
# - https://youtube.com/embed/영상ID
#
# ?si=... 등의 추가 주소 정보는 영상 ID에 포함하지 않습니다.
# ---------------------------------------------------------
def extract_video_id(url: str) -> str | None:
    """유튜브 주소에서 11자리 영상 ID를 추출합니다."""

    if not url:
        return None

    # 사용자가 주소 앞뒤에 입력한 공백을 제거합니다.
    url = url.strip()

    # http:// 또는 https://를 생략한 경우 https://를 붙입니다.
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname.lower() if parsed_url.hostname else ""
    except ValueError:
        return None

    video_id = None

    # youtu.be 형식의 짧은 주소 처리
    # 예: https://youtu.be/d95J8yzvjbQ?si=...
    if hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed_url.path.strip("/").split("/")[0]

    # youtube.com 형식의 주소 처리
    elif hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    ):
        path = parsed_url.path.rstrip("/")

        # 일반 영상 주소
        # 예: https://www.youtube.com/watch?v=d95J8yzvjbQ
        if path == "/watch":
            query_values = parse_qs(parsed_url.query)
            video_id = query_values.get("v", [None])[0]

        # Shorts 주소
        # 예: https://www.youtube.com/shorts/d95J8yzvjbQ
        elif path.startswith("/shorts/"):
            path_parts = path.split("/")
            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 영상 삽입 주소
        # 예: https://www.youtube.com/embed/d95J8yzvjbQ
        elif path.startswith("/embed/"):
            path_parts = path.split("/")
            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 이전 형식의 주소
        # 예: https://www.youtube.com/v/d95J8yzvjbQ
        elif path.startswith("/v/"):
            path_parts = path.split("/")
            if len(path_parts) >= 3:
                video_id = path_parts[2]

    # 유튜브 영상 ID는 보통 영문자, 숫자, -, _로 구성된 11자리입니다.
    if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return video_id

    return None


# ---------------------------------------------------------
# 4. YouTube API 오류 메시지를 확인하는 함수
# ---------------------------------------------------------
def get_youtube_error_reason(response: requests.Response) -> tuple[str, str]:
    """
    YouTube API가 보낸 오류 응답에서
    오류 사유와 상세 메시지를 꺼냅니다.
    """

    try:
        error_data = response.json()
    except ValueError:
        return "", ""

    error_object = error_data.get("error", {})
    error_message = error_object.get("message", "")

    errors = error_object.get("errors", [])

    if errors:
        error_reason = errors[0].get("reason", "")
    else:
        error_reason = ""

    return error_reason, error_message


# ---------------------------------------------------------
# 5. YouTube 댓글을 가져오는 함수
# ---------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_youtube_comments(
    video_id: str,
    api_key: str,
) -> pd.DataFrame:
    """
    YouTube Data API v3에서 최상위 댓글을 최대 100개 가져옵니다.

    API에서는 relevance 순으로 요청하고,
    받아온 뒤 좋아요 수 기준으로 다시 정렬합니다.
    """

    api_url = "https://www.googleapis.com/youtube/v3/commentThreads"

    # YouTube API에 전달할 요청 변수입니다.
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
            api_url,
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
            "인터넷 연결 상태를 확인한 뒤 다시 시도해 주세요."
        ) from error

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            "YouTube API를 요청하는 중 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        ) from error

    # HTTP 상태 코드가 정상 범위가 아니면 오류 내용을 확인합니다.
    if not response.ok:
        error_reason, error_message = get_youtube_error_reason(response)

        if error_reason == "commentsDisabled":
            raise RuntimeError(
                "이 영상은 댓글 사용이 중지되어 있어 댓글을 가져올 수 없습니다."
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
                "오늘 사용할 수 있는 YouTube API 할당량을 모두 사용했습니다. "
                "Google Cloud의 API 할당량을 확인해 주세요."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "댓글에 접근할 수 없습니다. "
                "댓글이 제한되었거나 API 설정에 문제가 있을 수 있습니다."
            )

        if response.status_code == 404:
            raise RuntimeError(
                "영상을 찾을 수 없습니다. 유튜브 링크를 다시 확인해 주세요."
            )

        # 위에서 구분하지 못한 오류는 상태 코드와 함께 안내합니다.
        detail = f" 상세 내용: {error_message}" if error_message else ""

        raise RuntimeError(
            f"YouTube API 요청에 실패했습니다. "
            f"오류 코드: {response.status_code}.{detail}"
        )

    # 정상 응답을 JSON 자료로 변환합니다.
    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "YouTube 서버에서 올바르지 않은 형식의 응답을 받았습니다."
        ) from error

    items = result.get("items", [])

    # 댓글이 하나도 없을 때 빈 데이터프레임을 반환합니다.
    if not items:
        return pd.DataFrame(
            columns=["댓글", "좋아요 수"]
        )

    comments = []

    # commentThreads 응답에서 최상위 댓글 정보를 하나씩 꺼냅니다.
    for item in items:
        try:
            top_level_comment = (
                item["snippet"]["topLevelComment"]["snippet"]
            )

            # 댓글 원문은 textOriginal 필드를 사용합니다.
            comment_text = top_level_comment.get("textOriginal", "")

            # 좋아요 수는 likeCount 필드를 사용합니다.
            # 혹시 문자열로 들어와도 숫자로 바꿀 수 있도록 처리합니다.
            like_count = pd.to_numeric(
                top_level_comment.get("likeCount", 0),
                errors="coerce",
            )

            if pd.isna(like_count):
                like_count = 0

            comments.append(
                {
                    "댓글": comment_text,
                    "좋아요 수": int(like_count),
                }
            )

        # 일부 댓글의 구조가 불완전하면 해당 댓글만 건너뜁니다.
        except (KeyError, TypeError):
            continue

    comments_df = pd.DataFrame(comments)

    if comments_df.empty:
        return pd.DataFrame(
            columns=["댓글", "좋아요 수"]
        )

    # 좋아요가 많은 댓글부터 보이도록 내림차순 정렬합니다.
    comments_df = comments_df.sort_values(
        by="좋아요 수",
        ascending=False,
    ).reset_index(drop=True)

    # 표에 순번을 추가합니다.
    comments_df.insert(
        0,
        "순번",
        range(1, len(comments_df) + 1),
    )

    return comments_df


# ---------------------------------------------------------
# 6. 화면 제목
# ---------------------------------------------------------
st.title("💬 유튜브 댓글 분석 앱")
st.caption(
    "1단계 · 유튜브 영상의 인기 댓글을 최대 100개 가져옵니다."
)


# ---------------------------------------------------------
# 7. 입력창의 현재 값을 세션 상태에 저장
#
# 세션 상태를 사용하면 예시 버튼을 눌렀을 때
# 입력창의 내용이 원하는 링크로 변경됩니다.
# ---------------------------------------------------------
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = EXAMPLE_URL_1


# ---------------------------------------------------------
# 8. 예시 링크 버튼
# ---------------------------------------------------------
st.subheader("예시 영상 선택")

example_column_1, example_column_2 = st.columns(2)

with example_column_1:
    if st.button(
        "예시 1 · 딥마인드 다큐(영어 댓글)",
        use_container_width=True,
    ):
        st.session_state.youtube_url = EXAMPLE_URL_1

with example_column_2:
    if st.button(
        "예시 2 · 2002 월드컵 추억(한국어 댓글)",
        use_container_width=True,
    ):
        st.session_state.youtube_url = EXAMPLE_URL_2


# ---------------------------------------------------------
# 9. 유튜브 링크 입력창
# ---------------------------------------------------------
video_url = st.text_input(
    "유튜브 영상 링크",
    key="youtube_url",
    placeholder="https://www.youtube.com/watch?v=...",
    help=(
        "youtu.be 짧은 주소와 youtube.com/watch 주소를 모두 사용할 수 있습니다."
    ),
)


# ---------------------------------------------------------
# 10. 분석 실행 버튼
# ---------------------------------------------------------
analyze_button = st.button(
    "댓글 가져오기",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# 11. 버튼을 눌렀을 때 댓글 가져오기
# ---------------------------------------------------------
if analyze_button:
    # 먼저 입력받은 링크에서 영상 ID를 추출합니다.
    video_id = extract_video_id(video_url)

    if video_id is None:
        st.error(
            "올바른 유튜브 영상 링크를 입력해 주세요. "
            "youtu.be 또는 youtube.com/watch 형식의 링크를 사용할 수 있습니다."
        )
        st.stop()

    # Streamlit Secrets에 API 키가 등록되어 있는지 확인합니다.
    try:
        youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
    except KeyError:
        st.error(
            "YouTube API 키가 등록되지 않았습니다. "
            "Streamlit Cloud의 Secrets에 "
            "`YOUTUBE_API_KEY`를 등록해 주세요."
        )

        st.code(
            'YOUTUBE_API_KEY = "발급받은_API_키"',
            language="toml",
        )

        st.stop()

    # 빈 문자열로 등록된 API 키도 확인합니다.
    if not str(youtube_api_key).strip():
        st.error(
            "등록된 YouTube API 키가 비어 있습니다. "
            "Streamlit Secrets 설정을 확인해 주세요."
        )
        st.stop()

    try:
        with st.spinner("유튜브 댓글을 가져오는 중입니다..."):
            comments_df = fetch_youtube_comments(
                video_id=video_id,
                api_key=str(youtube_api_key).strip(),
            )

    except RuntimeError as error:
        st.error(str(error))
        st.info(
            "영상 링크, 댓글 공개 여부, API 키와 API 할당량을 확인해 주세요."
        )
        st.stop()

    except Exception:
        # 예상하지 못한 내부 오류의 세부 내용은 사용자 화면에 노출하지 않습니다.
        st.error(
            "댓글을 처리하는 중 예상하지 못한 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )
        st.stop()

    # API 요청은 성공했지만 댓글이 없는 경우입니다.
    if comments_df.empty:
        st.warning(
            "가져올 수 있는 댓글이 없습니다. "
            "댓글이 없는 영상이거나 댓글 공개가 제한되었을 수 있습니다."
        )
        st.stop()

    st.success("댓글을 성공적으로 가져왔습니다.")

    # 확인용으로 추출된 영상 ID를 보여줍니다.
    st.caption(f"추출된 영상 ID: `{video_id}`")

    # 가져온 댓글 개수를 큰 지표 카드로 표시합니다.
    st.metric(
        label="가져온 댓글 수",
        value=f"{len(comments_df):,}개",
    )

    st.subheader("👍 좋아요가 많은 댓글")

    # 댓글과 좋아요 수를 표로 표시합니다.
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

else:
    st.info(
        "유튜브 영상 링크를 입력한 뒤 "
        "‘댓글 가져오기’ 버튼을 눌러 주세요."
    )
