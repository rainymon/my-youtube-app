import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from wordcloud import WordCloud


# ---------------------------------------------------------
# 1. Streamlit 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="유튜브 댓글 분석",
    page_icon="💬",
    layout="wide",
)


# ---------------------------------------------------------
# 2. 기본 설정값
# ---------------------------------------------------------
EXAMPLE_URL_1 = "https://youtu.be/d95J8yzvjbQ?si=LfL5DLwCL8Pk077r"
EXAMPLE_URL_2 = "https://youtu.be/I9vK5EVTt0U?si=NEZ8L7MRuNvrzINa"

YOUTUBE_API_URL = (
    "https://www.googleapis.com/youtube/v3/commentThreads"
)

# 한글 워드클라우드를 위한 나눔고딕 폰트 주소입니다.
FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/"
    "main/ofl/nanumgothic/NanumGothic-Regular.ttf"
)

# Streamlit Cloud에서도 파일을 저장할 수 있는 임시 폴더를 사용합니다.
FONT_PATH = Path("/tmp/NanumGothic-Regular.ttf")


# ---------------------------------------------------------
# 3. 유튜브 링크에서 영상 ID 추출
#
# 지원하는 주소 예시:
# - https://youtu.be/영상ID
# - https://www.youtube.com/watch?v=영상ID
# - https://youtube.com/shorts/영상ID
# - https://youtube.com/embed/영상ID
#
# ?si=... 같은 추가 정보는 자동으로 무시됩니다.
# ---------------------------------------------------------
def extract_video_id(url: str) -> str | None:
    """유튜브 주소에서 11자리 영상 ID를 추출합니다."""

    if not url:
        return None

    url = url.strip()

    # 사용자가 https://를 생략했을 때 자동으로 추가합니다.
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

    # youtu.be 짧은 주소
    if hostname in ("youtu.be", "www.youtu.be"):
        video_id = parsed_url.path.strip("/").split("/")[0]

    # youtube.com 주소
    elif hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    ):
        path = parsed_url.path.rstrip("/")

        # 일반 유튜브 영상 주소
        if path == "/watch":
            query_values = parse_qs(parsed_url.query)
            video_id = query_values.get("v", [None])[0]

        # 유튜브 Shorts 주소
        elif path.startswith("/shorts/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 임베드 주소
        elif path.startswith("/embed/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

        # 이전 형식의 영상 주소
        elif path.startswith("/v/"):
            path_parts = path.split("/")

            if len(path_parts) >= 3:
                video_id = path_parts[2]

    # 유튜브 영상 ID는 영문, 숫자, -, _로 구성된 11자리입니다.
    if video_id and re.fullmatch(
        r"[A-Za-z0-9_-]{11}",
        video_id,
    ):
        return video_id

    return None


# ---------------------------------------------------------
# 4. YouTube API 오류 내용 추출
# ---------------------------------------------------------
def get_youtube_error_reason(
    response: requests.Response,
) -> tuple[str, str]:
    """YouTube API 오류 응답에서 이유와 메시지를 추출합니다."""

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
# 5. YouTube 댓글 불러오기
#
# API에서는 relevance 순으로 최대 100개를 요청합니다.
# 받아온 댓글은 앱에서 좋아요 수가 많은 순으로 다시 정렬합니다.
# ---------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_youtube_comments(
    video_id: str,
    api_key: str,
) -> pd.DataFrame:
    """YouTube Data API에서 최상위 댓글을 최대 100개 가져옵니다."""

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

    # 정상 응답이 아니면 YouTube가 전달한 오류를 확인합니다.
    if not response.ok:
        error_reason, error_message = get_youtube_error_reason(
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
            f"YouTube API 요청에 실패했습니다. "
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

    for item in items:
        try:
            comment_snippet = (
                item["snippet"]["topLevelComment"]["snippet"]
            )

            # 댓글 원문은 textOriginal을 사용합니다.
            comment_text = comment_snippet.get(
                "textOriginal",
                "",
            )

            # 좋아요 수는 likeCount를 사용합니다.
            like_count = pd.to_numeric(
                comment_snippet.get("likeCount", 0),
                errors="coerce",
            )

            if pd.isna(like_count):
                like_count = 0

            comments.append(
                {
                    "댓글": str(comment_text),
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

    # 좋아요가 많은 댓글부터 정렬합니다.
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


# ---------------------------------------------------------
# 6. 댓글 전체에서 단어 추출
#
# - 한글과 영어 단어를 추출합니다.
# - 영어 단어는 모두 소문자로 통일합니다.
# - 한 글자짜리 단어는 제외합니다.
# - URL에 포함된 단어가 통계에 섞이지 않도록 주소를 먼저 제거합니다.
# ---------------------------------------------------------
def extract_words(comment_series: pd.Series) -> list[str]:
    """댓글 전체를 단어 목록으로 변환합니다."""

    all_comments = " ".join(
        comment_series.fillna("").astype(str).tolist()
    )

    # 댓글에 포함된 인터넷 주소를 제거합니다.
    all_comments = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        all_comments,
    )

    # 한글 또는 영어로 구성된 단어를 찾습니다.
    words = re.findall(
        r"[가-힣]+|[A-Za-z]+",
        all_comments,
    )

    cleaned_words = []

    for word in words:
        # 영어는 대소문자를 구분하지 않도록 소문자로 바꿉니다.
        normalized_word = word.lower()

        # 한 글자짜리 단어는 제외합니다.
        if len(normalized_word) <= 1:
            continue

        cleaned_words.append(normalized_word)

    return cleaned_words


# ---------------------------------------------------------
# 7. 단어 빈도 상위 20개 데이터 만들기
# ---------------------------------------------------------
def make_word_frequency_df(
    words: list[str],
) -> pd.DataFrame:
    """단어별 출현 횟수를 계산하고 상위 20개를 반환합니다."""

    word_counter = Counter(words)
    top_words = word_counter.most_common(20)

    return pd.DataFrame(
        top_words,
        columns=["단어", "빈도"],
    )


# ---------------------------------------------------------
# 8. 한글 폰트 내려받기
#
# 한 번 받은 폰트 파일은 캐시를 이용해 재사용합니다.
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_korean_font() -> str:
    """워드클라우드에 사용할 나눔고딕 폰트를 내려받습니다."""

    # 이미 폰트 파일이 있고 크기가 0보다 크면 다시 받지 않습니다.
    if FONT_PATH.exists() and FONT_PATH.stat().st_size > 0:
        return str(FONT_PATH)

    try:
        response = requests.get(
            FONT_URL,
            timeout=30,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "한글 폰트 다운로드 시간이 초과되었습니다. "
            "잠시 후 다시 시도해 주세요."
        ) from error

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            "한글 폰트를 내려받지 못했습니다. "
            "인터넷 연결 상태를 확인한 뒤 다시 시도해 주세요."
        ) from error

    if not response.content:
        raise RuntimeError(
            "내려받은 한글 폰트 파일이 비어 있습니다."
        )

    try:
        FONT_PATH.write_bytes(response.content)
    except OSError as error:
        raise RuntimeError(
            "한글 폰트 파일을 서버에 저장하지 못했습니다."
        ) from error

    return str(FONT_PATH)


# ---------------------------------------------------------
# 9. 워드클라우드 이미지 만들기
#
# matplotlib은 사용하지 않습니다.
# WordCloud가 만들어 주는 PIL 이미지를 st.image로 바로 표시합니다.
# ---------------------------------------------------------
def create_wordcloud_image(
    word_frequencies: dict[str, int],
    font_path: str,
):
    """단어 빈도 자료로 흰색 배경의 워드클라우드를 만듭니다."""

    wordcloud = WordCloud(
        font_path=font_path,
        width=1400,
        height=700,
        background_color="white",
        max_words=200,
        collocations=False,
    )

    wordcloud.generate_from_frequencies(word_frequencies)

    # matplotlib 없이 PIL 이미지로 변환합니다.
    return wordcloud.to_image()


# ---------------------------------------------------------
# 10. 화면 제목
# ---------------------------------------------------------
st.title("💬 유튜브 댓글 분석 앱")
st.caption(
    "유튜브 인기 댓글 수집, 단어 빈도 분석, 워드클라우드 만들기"
)


# ---------------------------------------------------------
# 11. 입력창의 초기값 설정
# ---------------------------------------------------------
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = EXAMPLE_URL_1


# ---------------------------------------------------------
# 12. 예시 영상 버튼
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
# 13. 유튜브 주소 입력창
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
# 14. 댓글 분석 버튼
# ---------------------------------------------------------
analyze_button = st.button(
    "댓글 분석하기",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# 15. 댓글 가져오기 및 분석
# ---------------------------------------------------------
if analyze_button:
    video_id = extract_video_id(video_url)

    if video_id is None:
        st.error(
            "올바른 유튜브 영상 링크를 입력해 주세요. "
            "youtu.be 또는 youtube.com/watch 형식을 사용할 수 있습니다."
        )
        st.stop()

    # Secrets에서 YouTube API 키를 불러옵니다.
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

    youtube_api_key = str(youtube_api_key).strip()

    if not youtube_api_key:
        st.error(
            "등록된 YouTube API 키가 비어 있습니다. "
            "Secrets 설정을 확인해 주세요."
        )
        st.stop()

    try:
        with st.spinner(
            "유튜브 댓글을 가져오는 중입니다..."
        ):
            comments_df = fetch_youtube_comments(
                video_id=video_id,
                api_key=youtube_api_key,
            )

    except RuntimeError as error:
        st.error(str(error))
        st.info(
            "영상 링크, 댓글 공개 여부, API 키와 "
            "API 할당량을 확인해 주세요."
        )
        st.stop()

    except Exception:
        st.error(
            "댓글을 처리하는 중 예상하지 못한 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )
        st.stop()

    if comments_df.empty:
        st.warning(
            "가져올 수 있는 댓글이 없습니다. "
            "댓글이 없거나 댓글 공개가 제한되었을 수 있습니다."
        )
        st.stop()

    st.success("댓글을 성공적으로 가져왔습니다.")
    st.caption(f"추출된 영상 ID: `{video_id}`")

    # -----------------------------------------------------
    # 16. 댓글 개수 지표 카드
    # -----------------------------------------------------
    st.metric(
        label="가져온 댓글 수",
        value=f"{len(comments_df):,}개",
    )

    # -----------------------------------------------------
    # 17. 댓글 목록 표
    # -----------------------------------------------------
    st.subheader("👍 좋아요가 많은 댓글")

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
    # 18. 댓글 전체를 단어로 나누기
    # -----------------------------------------------------
    words = extract_words(comments_df["댓글"])

    if not words:
        st.warning(
            "댓글에서 분석할 수 있는 두 글자 이상의 단어를 "
            "찾지 못했습니다."
        )
        st.stop()

    frequency_df = make_word_frequency_df(words)
    all_word_frequencies = dict(Counter(words))

    # -----------------------------------------------------
    # 19. 자주 나온 단어 상위 20개 막대그래프
    # -----------------------------------------------------
    st.subheader("📊 자주 나온 단어 상위 20개")
    st.caption(
        "한 글자짜리 단어는 제외했으며, "
        "영어는 대소문자를 구분하지 않습니다."
    )

    # 가로 그래프에서 가장 많이 나온 단어가 위쪽에 오도록
    # 빈도가 적은 단어부터 많은 단어 순으로 다시 정렬합니다.
    chart_df = frequency_df.sort_values(
        by="빈도",
        ascending=True,
    )

    word_frequency_figure = px.bar(
        chart_df,
        x="빈도",
        y="단어",
        orientation="h",
        text="빈도",
        labels={
            "단어": "단어",
            "빈도": "등장 횟수",
        },
        title="댓글에서 자주 나온 단어",
    )

    word_frequency_figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "등장 횟수: %{x:,}회"
            "<extra></extra>"
        ),
    )

    word_frequency_figure.update_layout(
        yaxis_title=None,
        xaxis_title="등장 횟수",
        xaxis_tickformat=",",
        margin={
            "l": 20,
            "r": 50,
            "t": 60,
            "b": 20,
        },
    )

    st.plotly_chart(
        word_frequency_figure,
        use_container_width=True,
    )

    # 상위 단어를 표로도 확인할 수 있게 표시합니다.
    with st.expander("상위 20개 단어를 표로 보기"):
        st.dataframe(
            frequency_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "단어": st.column_config.TextColumn(
                    "단어",
                    width="large",
                ),
                "빈도": st.column_config.NumberColumn(
                    "등장 횟수",
                    format="%,d회",
                ),
            },
        )

    st.divider()

    # -----------------------------------------------------
    # 20. 워드클라우드
    # -----------------------------------------------------
    st.subheader("☁️ 댓글 워드클라우드")
    st.caption(
        "댓글 전체에서 한 글자짜리 단어를 제외해 만들었습니다."
    )

    try:
        with st.spinner(
            "한글 폰트를 준비하고 워드클라우드를 만드는 중입니다..."
        ):
            font_path = download_korean_font()

            wordcloud_image = create_wordcloud_image(
                word_frequencies=all_word_frequencies,
                font_path=font_path,
            )

    except RuntimeError as error:
        st.error(str(error))
        st.info(
            "폰트 서버 연결 상태를 확인한 뒤 다시 실행해 주세요."
        )
        st.stop()

    except Exception:
        st.error(
            "워드클라우드 이미지를 만드는 중 문제가 발생했습니다."
        )
        st.stop()

    # matplotlib을 사용하지 않고 이미지를 바로 화면에 표시합니다.
    st.image(
        wordcloud_image,
        caption="댓글 전체 워드클라우드",
        use_container_width=True,
    )

else:
    st.info(
        "유튜브 영상 링크를 입력한 뒤 "
        "‘댓글 분석하기’ 버튼을 눌러 주세요."
    )
