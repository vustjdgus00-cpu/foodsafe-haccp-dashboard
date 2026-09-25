"""FOODSAFE: source-grounded HACCP food-safety exploration prototype."""
from __future__ import annotations

from pathlib import Path
import os
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="FOODSAFE", page_icon=":material/health_and_safety:", layout="wide")
st.html("""
<style>
.stMainBlockContainer { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
@media (max-width: 768px) {
  .stMainBlockContainer { padding-left: 1rem; padding-right: 1rem; padding-top: 1rem; }
}
</style>
""")

DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent / "data"
SOURCE_DIR = Path(os.getenv("FOODSAFE_DATA_DIR", str(DEFAULT_SOURCE_DIR)))
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

SOURCES = {
    "food_haccp": "20260914_114125.csv",
    "livestock_haccp": "20260916_100445.csv",
    "administrative": "식품검사 부적합 이력 + 개선 시정조치, 후속관리.csv",
    "nonconforming": "20260914_044911.csv",
    "recalls": "20260914_045015.csv",
    "geocoded": "20260918_044135.csv",
}
PROCESSED_SOURCES = {
    "food_haccp": "food_haccp_prepared.csv",
    "livestock_haccp": "livestock_haccp_prepared.csv",
    "administrative": "administrative_prepared.csv",
    "nonconforming": "nonconforming_prepared.csv",
    "recalls": "recalls_prepared.csv",
    "geocoded": "geocoded_prepared.csv",
}


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    path = SOURCE_DIR / SOURCES[name]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def read_processed(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def load_dataset(name: str) -> pd.DataFrame:
    """Prefer deployment-ready processed data, then fall back to source CSVs."""
    prepared = read_processed(PROCESSED_SOURCES[name])
    return prepared if not prepared.empty else read_csv(name)


def date_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    copy = frame.copy()
    if column in copy:
        copy[column] = pd.to_datetime(copy[column], errors="coerce")
    return copy


def region_series(frame: pd.DataFrame, address_columns: tuple[str, ...]) -> pd.Series:
    """Return a display region whether the prepared or raw schema is loaded."""
    if "sido" in frame.columns:
        return frame["sido"].fillna("주소 미상")
    for column in address_columns:
        if column in frame.columns:
            return frame[column].fillna("주소 미상").str.split().str[0].fillna("주소 미상")
    return pd.Series("주소 미상", index=frame.index, dtype="object")


def guide_response(prompt: str) -> str:
    """Answer product-use questions without representing this helper as a model prediction."""
    message = prompt.replace(" ", "").lower()
    if any(word in message for word in ("검색", "업체", "회사", "상호")):
        return "업체 통합 조회에서 등록 상호의 일부 2~4글자 또는 허가번호를 입력하세요. 결과에서 업체를 고르면 같은 허가번호로 정확히 연결된 HACCP 인증·부적합 검사·회수·행정처분 기록을 시간순으로 볼 수 있습니다."
    if any(word in message for word in ("지도", "지역", "시도", "시군구")):
        return "지역·지도 분석에서 시도, 시군구, 업종을 선택하세요. 지도는 HACCP 지정번호와 좌표가 정확히 연결된 인증만 표시하므로 전체 HACCP 업체 분포와는 다를 수 있습니다."
    if any(word in message for word in ("회수", "판매중지", "검사연결")):
        return "전체 회수 현황은 회수 분석에서, 부적합 검사와 같은 사건번호로 연결된 사례는 검사 → 회수 연결에서 보세요. 직접 연결 40건은 사건 흐름 분석 대상이며 전체 검사의 회수율은 아닙니다."
    if any(word in message for word in ("인증", "만료", "종료일", "생애주기")):
        return "인증 생애주기에서 종료일 기준 상태와 인증기간 분포를 확인하세요. '종료일 경과'는 위반·폐업·미인증을 자동으로 의미하지 않습니다."
    if any(word in message for word in ("그래프", "차트", "도표", "시각화")):
        return "맞춤 분석 메뉴에서 분석 주제와 보고 싶은 그래프를 선택하세요. 현재 데이터가 뒷받침하는 업종·지역·연도·품목·처분유형 그래프만 제공합니다."
    if any(word in message for word in ("후보", "위험", "점수", "예측")):
        return "우선 확인 후보는 위험 예측이나 위반 확정 목록이 아닙니다. HACCP 인증 명부와 행정처분 기록이 허가번호로 실제 연결된 추가 검토 대상입니다."
    if any(word in message for word in ("데이터", "출처", "한계")):
        return "정제본을 우선 사용하고, 없을 때만 원본 CSV를 읽습니다. 검사 부적합·회수 데이터는 업체 식별키 결측이 많아 업체별 위험점수에 사용하지 않습니다."
    return "무엇을 찾고 싶은지 말씀해 주세요. 예를 들어 ‘업체 검색 방법’, ‘어떤 그래프를 볼 수 있나요’, ‘후보는 무슨 뜻인가요’처럼 입력하면 안내합니다."


def show_source_notice() -> None:
    missing = [
        name
        for name, filename in SOURCES.items()
        if not (SOURCE_DIR / filename).exists() and not (PROCESSED_DIR / PROCESSED_SOURCES[name]).exists()
    ]
    if missing:
        st.warning("일부 데이터셋을 찾지 못했습니다: " + ", ".join(missing))
    st.caption("정제본을 우선 사용하며, 정제본이 없을 때만 원본 CSV를 읽습니다. 원본 데이터는 이 앱에서 수정하지 않습니다.")


DISPLAY_LABELS = {
    "LCNS_NO": "허가번호",
    "HACCP_APPN_NO": "HACCP 지정번호",
    "HACCP_APPN_DT": "HACCP 지정일",
    "appointno": "HACCP 지정번호",
    "BSSH_NM": "업체명",
    "BSSHNM": "업체명",
    "PRCSCITYPOINT_BSSHNM": "처분 시점 업체명",
    "INDUTY_CD_NM": "업종",
    "PRDLST_NM": "품목",
    "PRDLST_CD_NM": "품목유형",
    "PRDTNM": "제품명",
    "SITE_ADDR": "소재지",
    "ADDR": "주소",
    "sido": "시도",
    "sigungu": "시군구",
    "area1": "시도",
    "area2": "시군구",
    "source_type": "인증 구분",
    "businessitemNm": "업종·품목",
    "designation_date": "지정일",
    "certificate_end_date": "인증 종료일",
    "closure_date": "폐업일",
    "return_date": "인증 반납일",
    "certificate_duration_days": "인증기간(일)",
    "end_date_status": "종료일 기준 상태",
    "CRET_DTM": "기록일",
    "TEST_ITMNM": "검사 항목",
    "STDR_STND": "기준·규격",
    "TESTANALS_RSLT": "검사 결과",
    "INSTT_NM": "검사기관",
    "RTRVLDSUSE_SEQ": "사건번호",
    "inspection_record_date": "검사 기록일",
    "inspection_product": "검사 제품명",
    "inspection_items": "검사 항목",
    "inspection_institution": "검사기관",
    "recall_record_date": "회수 기록일",
    "recall_reason": "회수 사유",
    "recall_grade": "회수 등급",
    "recall_method": "회수 방법",
    "RTRVLPRVNS": "회수 사유",
    "RTRVL_GRDCD_NM": "회수 등급",
    "RTRVLPLANDOC_RTRVLMTHD": "회수 방법",
    "DSPS_DCSNDT": "처분 결정일",
    "DSPS_TYPECD_NM": "처분 유형",
    "LAWORD_CD_NM": "적용 법령",
    "DSPSCN": "처분 내용",
    "DSPS_INSTTCD_NM": "처분기관",
    "dataset": "데이터셋",
    "raw_rows": "원본 행 수",
    "prepared_rows": "정제 행 수",
    "removed_exact_duplicates": "제거한 완전 중복",
    "raw_exact_duplicates": "원본 완전 중복",
    "key": "핵심 식별키",
    "key_unique_non_null": "고유 키 수",
    "key_non_null_rows": "키 보유 행",
    "valid_date_rows": "유효 날짜 행",
    "address_non_null_rows": "주소 보유 행",
    "licence_present_rows": "허가번호 보유 행",
    "unique_licences": "고유 허가번호",
    "rows_exactly_linked_to_food_haccp": "식품 HACCP 정확 연결 행",
    "rows_exactly_linked_to_any_haccp": "전체 HACCP 정확 연결 행",
    "rule": "연결 규칙",
}


def show_dataframe(frame: pd.DataFrame, **kwargs: object) -> None:
    """Display user-facing Korean labels without changing source columns."""
    view = frame.copy()
    date_columns = {
        "HACCP_APPN_DT", "designation_date", "certificate_end_date", "closure_date", "return_date",
        "CRET_DTM", "inspection_record_date", "recall_record_date", "DSPS_DCSNDT",
    }
    for column in date_columns.intersection(view.columns):
        parsed = pd.to_datetime(view[column], errors="coerce")
        view[column] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), view[column])
    st.dataframe(view.rename(columns=DISPLAY_LABELS), **kwargs)


@st.cache_data(show_spinner=False)
def read_manifest() -> dict[str, object]:
    path = PROCESSED_DIR / "data_quality_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


food = load_dataset("food_haccp")
livestock = load_dataset("livestock_haccp")
administrative = date_column(load_dataset("administrative"), "DSPS_DCSNDT")
nonconforming = date_column(load_dataset("nonconforming"), "CRET_DTM")
recalls = date_column(load_dataset("recalls"), "CRET_DTM")
geocoded = load_dataset("geocoded")
prepared_candidates = read_processed("priority_check_candidates.csv")
linkage_audit = read_processed("record_linkage_audit.csv")
haccp_lifecycle = read_processed("haccp_lifecycle.csv")
inspection_recall_events = read_processed("inspection_recall_events.csv")
haccp_geocoded = read_processed("haccp_geocoded.csv")
manifest = read_manifest()

st.sidebar.title("FOODSAFE")
st.sidebar.caption("HACCP 식품안전 우선 확인 지원")
page = st.sidebar.radio(
    "메뉴",
    [
        "종합 현황",
        "인증 생애주기",
        "검사·부적합",
        "검사 → 회수 연결",
        "회수 분석",
        "행정처분 분석",
        "지역·지도 분석",
        "업체 통합 조회",
        "우선 확인 후보",
        "데이터 품질",
        "데이터·분석 방법",
    ],
    key="navigation_menu",
)
st.sidebar.caption("분석 결과는 공개된 관측 기록을 요약하며, 개별 업체의 위험을 판정하지 않습니다.")
show_source_notice()

page_guides = {
    "HACCP 인증 분석": ("인증 명부가 어떤 지역·업종·품목으로 구성되어 있는가?", "지역과 업종을 선택하면 지표·차트·명부가 함께 바뀌니다.", "인증 행 수와 고유 업체 수는 서로 다릅니다."),
    "인증 생애주기": ("인증 종료일을 기준으로 어떤 상태가 많은가?", "대상과 상태를 고른 뒤 종료일 구성과 인증기간 분포를 비교하세요.", "'종료일 경과'는 위반이나 현재 영업 여부를 뜻하지 않습니다."),
    "검사·부적합": ("어떤 검사 항목·품목·기관의 기록이 많은가?", "시도와 품목을 선택하고 상위 검사 항목과 기관×항목 격자를 읽으세요.", "기록 수는 검사 빈도와 공개 범위의 영향을 받으며 위험도가 아닙니다."),
    "검사 → 회수 연결": ("같은 사건번호로 검사와 회수가 연결된 사례는 어떻게 흐르는가?", "품목·검사항목·회수사유·회수방법의 흐름을 좌→우로 보세요.", "연결된 40개 고유 사건만 대상이며 전체 검사의 회수 전환율은 아닙니다."),
    "회수 분석": ("회수 기록은 어떤 품목·사유·방법에 집중되어 있는가?", "조건을 줄인 뒤 품목 면적과 등급×방법 격자를 함께 비교하세요.", "공개된 회수·판매중지 기록의 분포이며 시장 전체 발생률이 아닙니다."),
    "행정처분 분석": ("행정처분은 시간·지역·유형·법령별로 어떻게 분포하는가?", "지역·처분유형·업종 필터로 범위를 줄이고 연도 격자와 법령 구성을 보세요.", "모든 행정처분을 식품안전 부적합으로 해석하지 않습니다."),
    "지역·지도 분석": ("정확히 좌표가 연결된 HACCP 인증은 어디에 분포하는가?", "시도·시군구·업종을 선택하면 지도와 지역 순위가 함께 바뀌니다.", "지정번호로 정확 연결된 인증만 표시하므로 전체 명부를 대표하지 않습니다."),
    "업체 통합 조회": ("특정 HACCP 업체와 정확히 연결되는 기록은 무엇인가?", "상호 일부 또는 허가번호로 찾고 선택한 뒤 시간순 사건을 확인하세요.", "연결 기록이 없다고 안전을 의미하지 않으며, 상호·주소 유사도 매칭은 하지 않습니다."),
    "우선 확인 후보": ("인증 명부와 행정처분이 허가번호로 겹치는 대상은 무엇인가?", "필터로 범위를 줄이고 업체 상세에서 인증과 처분 원문 필드를 함께 확인하세요.", "후보는 추가 확인 대상이지 위험 판정·순위·예측 결과가 아닙니다."),
    "반복 사건 기록": ("같은 허가번호에 복수 사건이 관측된 경우는 무엇인가?", "사건 유형별 건수 격자와 원본 요약표를 함께 보세요.", "반복 기록은 공개 기록의 반복이며 현재 상태나 미래 위험을 의미하지 않습니다."),
}
def render_page_guide(page_name: str) -> None:
    if page_name not in page_guides:
        return
    question, action, caveat = page_guides[page_name]
    with st.expander("이 화면 읽는 법", icon=":material/help:"):
        st.markdown(f"**핵심 질문**  {question}\n\n**사용 방법**  {action}\n\n**해석 주의**  {caveat}")


if page == "종합 현황":
    st.title("FOODSAFE", icon=":material/health_and_safety:")
    st.badge("데이터 기반 프로토타입", icon=":material/verified:", color="blue")
    st.subheader("식품안전 통합 데이터 시각화 플랫폼")
    st.caption("HACCP 인증, 부적합 검사, 회수, 행정처분을 실제 식별키로 연결해 시간·공간·사건 흐름을 탐색합니다.")
    with st.container(border=True):
        narrative_left, narrative_center, narrative_right = st.columns(3)
        narrative_left.markdown("**데이터 연결**\n\n허가번호·인증번호·사건번호가 정확히 일치할 때만 연결합니다.")
        narrative_center.markdown("**분석 경험**\n\n인증 생애주기, 검사→회수 흐름, 지역 분포를 서로 다른 시각화로 탐색합니다.")
        narrative_right.markdown("**해석 원칙**\n\n위험을 판정하지 않고 관측 기록과 데이터 한계를 함께 보여줍니다.")
    with st.expander("처음 사용하는 방법", icon=":material/help:"):
        st.markdown("- **전체 현황이 궁금하면** 인증·검사·회수·행정처분 분석을 선택하세요.\n- **특정 업체를 찾으면** 업체 통합 조회에서 상호 일부 또는 허가번호를 검색하세요.\n- **지역 패턴이 궁금하면** 지역·지도 분석을 사용하세요.\n- **수치의 근거와 한계가 궁금하면** 데이터 품질과 데이터·분석 방법을 확인하세요.")
    home_metrics = st.columns(4)
    home_metrics[0].metric("식품 HACCP 업체", f"{food['LCNS_NO'].nunique():,}" if not food.empty else "자료 없음", border=True)
    home_metrics[1].metric("축산물 HACCP 업체", f"{livestock['LCNS_NO'].nunique():,}" if not livestock.empty else "자료 없음", border=True)
    home_metrics[2].metric("HACCP 고유 인증", f"{len(haccp_lifecycle):,}" if not haccp_lifecycle.empty else "자료 없음", border=True)
    home_metrics[3].metric("행정처분 기록", f"{len(administrative):,}", border=True)
    event_metrics = st.columns(4)
    event_metrics[0].metric("부적합 검사 기록", f"{len(nonconforming):,}", border=True)
    event_metrics[1].metric("회수·판매중지 기록", f"{len(recalls):,}", border=True)
    event_metrics[2].metric("검사–회수 직접 연결", f"{inspection_recall_events['RTRVLDSUSE_SEQ'].nunique():,}" if not inspection_recall_events.empty else "자료 없음", border=True)
    event_metrics[3].metric("우선 확인 후보", f"{prepared_candidates['LCNS_NO'].nunique():,}" if not prepared_candidates.empty else "자료 없음", border=True)

    overview_left, overview_right = st.columns(2)
    if not haccp_lifecycle.empty:
        with overview_left.container(border=True, height="stretch"):
            st.subheader("인증 종료일 기준 상태")
            st.caption("종료일 경과·90일 이내 종료 예정·이후 종료의 구성을 보여줍니다.")
            home_lifecycle = haccp_lifecycle.groupby(["source_type", "end_date_status"]).size().reset_index(name="인증 수")
            st.plotly_chart(px.bar(home_lifecycle, x="source_type", y="인증 수", color="end_date_status", barmode="stack", color_discrete_map={"종료일까지 91일 이상":"#168A8A", "90일 이내 종료 예정":"#F59E0B", "종료일 경과":"#D95F59"}), width="stretch")
    if not administrative.empty:
        with overview_right.container(border=True, height="stretch"):
            st.subheader("행정처분 유형")
            st.caption("공개된 처분 기록의 유형별 규모를 비교합니다.")
            home_admin = administrative["DSPS_TYPECD_NM"].fillna("미상").value_counts().head(10).rename_axis("처분 유형").reset_index(name="기록 수")
            st.plotly_chart(px.bar(home_admin, x="기록 수", y="처분 유형", orientation="h", color_discrete_sequence=["#D95F59"]), width="stretch")

    focus_left, focus_right = st.columns(2)
    with focus_left.container(border=True, height="stretch"):
        st.subheader("검사 → 회수 직접 연결")
        st.caption("동일 사건번호로 연결된 고유 사건만 집계합니다.")
        flow_summary = pd.DataFrame({"구분":["부적합 검사 전체", "회수 전체", "직접 연결 사건"], "기록 수":[len(nonconforming), len(recalls), inspection_recall_events["RTRVLDSUSE_SEQ"].nunique() if not inspection_recall_events.empty else 0]})
        st.plotly_chart(px.bar(flow_summary, x="구분", y="기록 수", color="구분", color_discrete_sequence=["#2563EB", "#D95F59", "#0F766E"]), width="stretch")
    with focus_right.container(border=True, height="stretch"):
        st.subheader("분석 바로가기")
        st.caption("심사 시연에서 데이터의 시간·흐름·공간·품질을 순서대로 확인할 수 있습니다.")
        st.markdown("**1. 인증 생애주기** — 지정부터 종료·반납까지\n\n**2. 검사 → 회수 연결** — 40개 직접 연결 사건\n\n**3. 지역·지도 분석** — 지정번호 기반 좌표 연결\n\n**4. 업체 통합 조회** — 허가번호 기반 사건 이력\n\n**5. 데이터 품질** — 결측·연결률·중복 공개")

elif page == "HACCP 인증 분석":
    st.title("HACCP 인증 분석", icon=":material/verified:")
    render_page_guide(page)
    selected = st.segmented_control("대상", ["식품", "축산물"], default="식품", key="haccp_target")
    source_frame = food if selected == "식품" else livestock
    frame = source_frame.copy()
    if frame.empty:
        st.info("원본 파일을 찾지 못했습니다.")
    else:
        frame["표시 시도"] = frame["SITE_ADDR"].fillna("주소 미상").str.split().str[0]
        with st.container(border=True):
            st.caption("원하는 지역 또는 업종을 고르면 아래 지표·차트·명부가 함께 바뀝니다. 선택하지 않으면 전체를 표시합니다.")
            filter_left, filter_right = st.columns(2)
            selected_regions = filter_left.multiselect("지역", sorted(frame["표시 시도"].dropna().unique()), key="haccp_regions")
            selected_industries = filter_right.multiselect("업종", sorted(frame["INDUTY_CD_NM"].dropna().unique()), key="haccp_industries")
        if selected_regions:
            frame = frame[frame["표시 시도"].isin(selected_regions)]
        if selected_industries:
            frame = frame[frame["INDUTY_CD_NM"].isin(selected_industries)]
        if frame.empty:
            st.info("현재 선택 조건에 해당하는 HACCP 명부 행이 없습니다. 필터를 조정해 보세요.")
            st.stop()
        comparison_regions = selected_regions or frame["표시 시도"].dropna().unique().tolist()
        admin_region = region_series(administrative, ("ADDR", "SITE_ADDR"))
        inspection_region = region_series(nonconforming, ("ADDR", "SITE_ADDR"))
        recall_region = region_series(recalls, ("ADDR", "SITE_ADDR"))
        candidate_region = prepared_candidates.get("haccp_sido", pd.Series(dtype=str)).fillna("주소 미상")
        regional_metrics = st.columns(5)
        regional_metrics[0].metric("HACCP 고유 허가번호", f"{frame['LCNS_NO'].nunique():,}", border=True)
        regional_metrics[1].metric("HACCP 지정번호", f"{frame['HACCP_APPN_NO'].nunique():,}", border=True)
        regional_metrics[2].metric("행정처분 기록", f"{administrative[admin_region.isin(comparison_regions)].shape[0]:,}", border=True)
        regional_metrics[3].metric("부적합 검사 기록", f"{nonconforming[inspection_region.isin(comparison_regions)].shape[0]:,}", border=True)
        regional_metrics[4].metric("회수·판매중지 기록", f"{recalls[recall_region.isin(comparison_regions)].shape[0]:,}", border=True)
        st.caption("행정처분·검사·회수는 주소 기준 지역 분포입니다. 같은 지역의 기록을 함께 보여줄 뿐, 업체 간 직접 인과관계를 뜻하지 않습니다.")

        regional_summary = pd.DataFrame({"시도": comparison_regions})
        regional_summary["HACCP 업체"] = regional_summary["시도"].map(frame.groupby("표시 시도")["LCNS_NO"].nunique()).fillna(0).astype(int)
        regional_summary["행정처분"] = regional_summary["시도"].map(admin_region.value_counts()).fillna(0).astype(int)
        regional_summary["부적합 검사"] = regional_summary["시도"].map(inspection_region.value_counts()).fillna(0).astype(int)
        regional_summary["회수·판매중지"] = regional_summary["시도"].map(recall_region.value_counts()).fillna(0).astype(int)
        if not prepared_candidates.empty:
            regional_summary["우선 확인 후보"] = regional_summary["시도"].map(candidate_region.value_counts()).fillna(0).astype(int)
        with st.expander("지역별 통합 지표표", icon=":material/table_chart:"):
            show_dataframe(regional_summary.sort_values("HACCP 업체", ascending=False), hide_index=True, width="stretch")

        region = frame["표시 시도"].value_counts().head(20).rename_axis("시도").reset_index(name="인증 행 수")
        chart_left, chart_right = st.columns(2)
        chart_left.plotly_chart(
            px.bar(region, x="시도", y="인증 행 수", title=f"{selected} HACCP 주소 첫 행정구역별 인증 행 수"),
            width="stretch",
        )
        hierarchy_columns = ["INDUTY_CD_NM"]
        if "PRDLST_NM" in frame.columns and frame["PRDLST_NM"].notna().any():
            hierarchy_columns.append("PRDLST_NM")
        hierarchy = (
            frame.assign(**{column: frame[column].fillna("미상") for column in hierarchy_columns})
            .groupby(hierarchy_columns)["HACCP_APPN_NO"]
            .nunique()
            .reset_index(name="HACCP 지정번호 수")
        )
        chart_right.plotly_chart(
            px.sunburst(hierarchy, path=hierarchy_columns, values="HACCP 지정번호 수", title=f"{selected} 업종·품목 구성"),
            width="stretch",
        )
        display_columns = [
            column
            for column in ["LCNS_NO", "BSSH_NM", "INDUTY_CD_NM", "PRDLST_NM", "HACCP_APPN_DT", "HACCP_APPN_NO", "SITE_ADDR"]
            if column in frame.columns
        ]
        with st.container(border=True):
            st.subheader("인증 명부 미리보기")
            show_dataframe(frame[display_columns].head(500), hide_index=True, width="stretch")
elif page == "인증 생애주기":
    st.title("인증 생애주기", icon=":material/timeline:")
    render_page_guide(page)
    st.caption("인증 지정부터 종료·반납·폐업까지의 시간적 변화를 보여줍니다. 상태는 법적 판정이 아니라 인증 종료일 기준입니다.")
    if haccp_lifecycle.empty:
        st.info("생애주기 정제 데이터가 없습니다. 전처리를 다시 실행해 주세요.")
    else:
        life = haccp_lifecycle.copy()
        for column in ["designation_date", "certificate_end_date", "closure_date", "return_date"]:
            life[column] = pd.to_datetime(life[column], errors="coerce")
        life["certificate_duration_days"] = pd.to_numeric(life["certificate_duration_days"], errors="coerce")
        filter_a, filter_b, filter_c = st.columns(3)
        selected_sources = filter_a.multiselect("대상", sorted(life["source_type"].dropna().unique()), key="lifecycle_sources")
        selected_statuses = filter_b.multiselect("종료일 기준 상태", sorted(life["end_date_status"].dropna().unique()), key="lifecycle_statuses")
        selected_life_regions = filter_c.multiselect("시도", sorted(life["sido"].dropna().unique()), key="lifecycle_regions")
        if selected_sources:
            life = life[life["source_type"].isin(selected_sources)]
        if selected_statuses:
            life = life[life["end_date_status"].isin(selected_statuses)]
        if selected_life_regions:
            life = life[life["sido"].isin(selected_life_regions)]
        status_counts = life["end_date_status"].value_counts()
        metrics = st.columns(4)
        metrics[0].metric("고유 인증", f"{life['HACCP_APPN_NO'].nunique():,}", border=True)
        metrics[1].metric("90일 이내 종료 예정", f"{status_counts.get('90일 이내 종료 예정', 0):,}", border=True)
        metrics[2].metric("종료일 경과", f"{status_counts.get('종료일 경과', 0):,}", border=True)
        metrics[3].metric("반납일 기록", f"{life['return_date'].notna().sum():,}", border=True)
        chart_left, chart_right = st.columns(2)
        with chart_left.container(border=True):
            st.subheader("종료일 기준 상태 구성")
            st.caption("선택한 인증이 종료일을 기준으로 어느 구간에 속하는지 비교합니다.")
            composition = life.groupby(["source_type", "end_date_status"]).size().reset_index(name="인증 수")
            st.plotly_chart(px.bar(composition, x="source_type", y="인증 수", color="end_date_status", barmode="stack", color_discrete_map={"종료일까지 91일 이상":"#168A8A", "90일 이내 종료 예정":"#F59E0B", "종료일 경과":"#D95F59"}), width="stretch")
        with chart_right.container(border=True):
            st.subheader("인증기간 분포")
            st.caption("지정일부터 인증 종료일까지의 기간을 대상별로 비교합니다.")
            duration = life[life["certificate_duration_days"].between(0, 5000)]
            st.plotly_chart(px.box(duration, x="source_type", y="certificate_duration_days", color="source_type", labels={"certificate_duration_days":"인증기간(일)", "source_type":"대상"}), width="stretch")
        end_month = life.dropna(subset=["certificate_end_date"]).assign(종료월=lambda x: x["certificate_end_date"].dt.to_period("M").astype(str)).groupby(["종료월", "end_date_status"]).size().reset_index(name="인증 수")
        with st.container(border=True):
            st.subheader("월별 인증 종료 분포")
            st.caption("인증 종료일이 집중되는 월과 상태 구성을 보여줍니다.")
            st.plotly_chart(px.area(end_month, x="종료월", y="인증 수", color="end_date_status", color_discrete_map={"종료일까지 91일 이상":"#168A8A", "90일 이내 종료 예정":"#F59E0B", "종료일 경과":"#D95F59"}), width="stretch")
        timeline_columns = ["source_type", "BSSH_NM", "INDUTY_CD_NM", "HACCP_APPN_NO", "designation_date", "certificate_end_date", "closure_date", "return_date", "end_date_status"]
        with st.expander("인증 상세 기록", icon=":material/table_chart:"):
            show_dataframe(life[timeline_columns].sort_values("certificate_end_date").head(1000), hide_index=True, width="stretch")
            st.download_button("현재 인증 결과 다운로드", life.to_csv(index=False).encode("utf-8-sig"), "foodsafe_lifecycle_filtered.csv", "text/csv", icon=":material/download:")

elif page == "검사·부적합":
    st.title("검사·부적합 분석", icon=":material/fact_check:")
    render_page_guide(page)
    inspection_view = nonconforming.copy()
    if not inspection_view.empty:
        inspection_view["표시 시도"] = region_series(inspection_view, ("ADDR",))
        with st.container(border=True):
            st.caption("기간·지역·품목·검사 항목을 선택하면 이 페이지의 부적합 분석이 함께 갱신됩니다. 업종 정보는 현재 검사 데이터에 직접 포함되어 있지 않습니다.")
            filter_a, filter_b, filter_c, filter_d = st.columns(4)
            selected_inspection_regions = filter_a.multiselect("지역", sorted(inspection_view["표시 시도"].dropna().unique()), key="inspection_regions")
            selected_products = filter_b.multiselect("품목", sorted(inspection_view["PRDLST_CD_NM"].dropna().unique()), key="inspection_products")
            selected_items = filter_c.multiselect("검사 항목", sorted(inspection_view["TEST_ITMNM"].dropna().unique()), key="inspection_items")
            valid_inspection_dates = inspection_view["CRET_DTM"].dropna()
            selected_inspection_dates = filter_d.date_input(
                "기록 기간",
                value=(valid_inspection_dates.min().date(), valid_inspection_dates.max().date()) if not valid_inspection_dates.empty else None,
                key="inspection_dates",
            )
        if isinstance(selected_inspection_dates, tuple) and len(selected_inspection_dates) == 2:
            inspection_view = inspection_view[
                inspection_view["CRET_DTM"].between(pd.Timestamp(selected_inspection_dates[0]), pd.Timestamp(selected_inspection_dates[1]))
            ]
        if selected_inspection_regions:
            inspection_view = inspection_view[inspection_view["표시 시도"].isin(selected_inspection_regions)]
        if selected_products:
            inspection_view = inspection_view[inspection_view["PRDLST_CD_NM"].isin(selected_products)]
        if selected_items:
            inspection_view = inspection_view[inspection_view["TEST_ITMNM"].isin(selected_items)]
    if not inspection_view.empty:
        inspection_metrics = st.columns(4)
        inspection_metrics[0].metric("부적합 기록", f"{len(inspection_view):,}", border=True)
        inspection_metrics[1].metric("검사 항목", f"{inspection_view['TEST_ITMNM'].nunique():,}", border=True)
        inspection_metrics[2].metric("품목", f"{inspection_view['PRDLST_CD_NM'].nunique():,}", border=True)
        inspection_metrics[3].metric("검사기관", f"{inspection_view['INSTT_NM'].nunique():,}", border=True)
        items = inspection_view["TEST_ITMNM"].fillna("미상").value_counts().head(15).rename_axis("검사 항목").reset_index(name="사례 수")
        product = inspection_view["PRDLST_CD_NM"].fillna("품목 미상").value_counts().head(20).rename_axis("품목").reset_index(name="사례 수")
        top_item_share = items.iloc[0]["사례 수"] / len(inspection_view) * 100
        st.caption(f"현재 조건에서 가장 많이 기록된 검사 항목은 **{items.iloc[0]['검사 항목']}**이며, 전체 표시 기록의 {top_item_share:.1f}%입니다.")
        inspection_left, inspection_right = st.columns(2)
        with inspection_left.container(border=True):
            st.subheader("부적합 검사 항목")
            st.caption("선택 조건에서 가장 많이 기록된 검사 항목을 비교합니다.")
            st.plotly_chart(px.bar(items, x="사례 수", y="검사 항목", orientation="h"), width="stretch")
        with inspection_right.container(border=True):
            st.subheader("부적합 품목 구성")
            st.caption("부적합 기록이 어떤 품목으로 구성되는지 면적으로 보여줍니다.")
            st.plotly_chart(px.treemap(product, path=["품목"], values="사례 수"), width="stretch")
        institution_item = inspection_view.assign(검사기관=inspection_view["INSTT_NM"].fillna("기관 미상"), 검사항목=inspection_view["TEST_ITMNM"].fillna("항목 미상"))
        top_institutions = institution_item["검사기관"].value_counts().head(15).index
        top_items = institution_item["검사항목"].value_counts().head(15).index
        institution_matrix = institution_item[institution_item["검사기관"].isin(top_institutions) & institution_item["검사항목"].isin(top_items)].groupby(["검사기관", "검사항목"]).size().unstack(fill_value=0)
        with st.container(border=True):
            st.subheader("검사기관·검사 항목 매트릭스")
            st.caption("기관별로 어떤 부적합 검사 항목이 기록됐는지 보여줍니다. 기관별 부적합률을 의미하지 않습니다.")
            st.plotly_chart(px.imshow(institution_matrix, aspect="auto", labels={"x":"검사 항목", "y":"검사기관", "color":"기록 수"}, color_continuous_scale="Teal"), width="stretch")
        with st.expander("부적합 검사 상세 기록", icon=":material/table_chart:"):
            show_dataframe(inspection_view[["CRET_DTM", "INSTT_NM", "PRDTNM", "PRDLST_CD_NM", "TEST_ITMNM", "STDR_STND", "TESTANALS_RSLT", "ADDR"]].sort_values("CRET_DTM", ascending=False).head(500), hide_index=True, width="stretch")
            st.download_button("현재 부적합 검사 결과 다운로드", inspection_view.to_csv(index=False).encode("utf-8-sig"), "foodsafe_nonconforming_filtered.csv", "text/csv", icon=":material/download:")
    st.info("부적합 검사와 회수 기록은 업체 식별키 결측이 많으므로, 업체 단위 위험점수에는 사용하지 않고 사건·항목 분석으로 제시합니다.")

elif page == "검사 → 회수 연결":
    st.title("검사 → 회수 연결", icon=":material/account_tree:")
    render_page_guide(page)
    st.caption("동일 사건 식별번호로 직접 연결된 부적합 검사와 회수 기록입니다. 여러 검사 항목은 하나의 사건으로 집계합니다.")
    if inspection_recall_events.empty:
        st.info("검사–회수 연결 데이터가 없습니다.")
    else:
        events = inspection_recall_events.copy()
        selected_grades = st.multiselect("회수등급", sorted(events["recall_grade"].dropna().unique()), key="flow_grades")
        if selected_grades:
            events = events[events["recall_grade"].isin(selected_grades)]
        metrics = st.columns(4)
        metrics[0].metric("직접 연결 사건", f"{events['RTRVLDSUSE_SEQ'].nunique():,}", border=True)
        metrics[1].metric("검사 항목", f"{events['inspection_items'].str.split(' | ', regex=False).explode().nunique():,}", border=True)
        metrics[2].metric("회수 사유", f"{events['recall_reason'].nunique():,}", border=True)
        metrics[3].metric("회수방법", f"{events['recall_method'].nunique():,}", border=True)
        flow = events.assign(
            검사항목=events["inspection_items"].str.split(" | ", regex=False).str[0].fillna("검사항목 미상"),
            회수사유=events["recall_reason"].fillna("사유 미상"),
            회수등급=events["recall_grade"].fillna("등급 미상"),
            회수방법=events["recall_method"].fillna("방법 미상"),
        )
        stages = [("검사항목", "회수사유"), ("회수사유", "회수등급"), ("회수등급", "회수방법")]
        labels = []
        for stage_name in ["검사항목", "회수사유", "회수등급", "회수방법"]:
            labels.extend([f"{stage_name}: {value}" for value in flow[stage_name].drop_duplicates()])
        label_index = {label: index for index, label in enumerate(labels)}
        sources, targets, values = [], [], []
        for left_name, right_name in stages:
            links = flow.groupby([left_name, right_name]).size().reset_index(name="사건 수")
            for row in links.itertuples(index=False):
                sources.append(label_index[f"{left_name}: {row[0]}"])
                targets.append(label_index[f"{right_name}: {row[1]}"])
                values.append(int(row[2]))
        sankey = go.Figure(go.Sankey(node={"label": labels, "pad": 14, "thickness": 15, "color": "#2B7A78"}, link={"source": sources, "target": targets, "value": values, "color": "rgba(43,122,120,0.25)"}))
        sankey.update_layout(height=680, margin={"l":10,"r":10,"t":20,"b":10})
        with st.container(border=True):
            st.subheader("부적합 검사에서 회수조치까지의 실제 흐름")
            st.caption("선의 굵기는 고유 사건 수입니다. 항목이 많은 사건은 대표 검사 항목으로 표시하고 전체 항목은 아래 표에서 확인합니다.")
            st.plotly_chart(sankey, width="stretch")
        with st.expander("직접 연결 사건 근거표", icon=":material/table_chart:"):
            show_dataframe(events[["RTRVLDSUSE_SEQ", "inspection_record_date", "inspection_product", "inspection_items", "inspection_institution", "recall_record_date", "recall_reason", "recall_grade", "recall_method"]], hide_index=True, width="stretch")
            st.download_button("현재 연결 사건 다운로드", events.to_csv(index=False).encode("utf-8-sig"), "foodsafe_inspection_recall_events.csv", "text/csv", icon=":material/download:")

elif page == "회수 분석":
    st.title("회수·판매중지 분석", icon=":material/assignment_return:")
    render_page_guide(page)
    st.caption("회수·판매중지 사건 기록을 시점·사유·품목 기준으로 탐색합니다. 업체 연결은 허가번호가 있는 기록에만 가능합니다.")
    if recalls.empty:
        st.info("회수·판매중지 데이터가 없습니다.")
    else:
        recall_view = recalls.copy()
        recall_view["표시 시도"] = region_series(recall_view, ("ADDR",))
        filter_left, filter_center, filter_right = st.columns(3)
        selected_recall_regions = filter_left.multiselect("지역", sorted(recall_view["표시 시도"].dropna().unique()), key="recall_regions")
        selected_recall_products = filter_center.multiselect("품목", sorted(recall_view["PRDLST_CD_NM"].dropna().unique()), key="recall_products")
        valid_recall_dates = recall_view["CRET_DTM"].dropna()
        selected_recall_dates = filter_right.date_input(
            "기록 기간",
            value=(valid_recall_dates.min().date(), valid_recall_dates.max().date()) if not valid_recall_dates.empty else None,
            key="recall_dates",
        )
        if isinstance(selected_recall_dates, tuple) and len(selected_recall_dates) == 2:
            recall_view = recall_view[
                recall_view["CRET_DTM"].between(pd.Timestamp(selected_recall_dates[0]), pd.Timestamp(selected_recall_dates[1]))
            ]
        if selected_recall_regions:
            recall_view = recall_view[recall_view["표시 시도"].isin(selected_recall_regions)]
        if selected_recall_products:
            recall_view = recall_view[recall_view["PRDLST_CD_NM"].isin(selected_recall_products)]
        if recall_view.empty:
            st.info("현재 선택 조건에 해당하는 회수 기록이 없습니다. 필터를 조정해 보세요.")
            st.stop()
        metrics = st.columns(3)
        metrics[0].metric("회수 기록", f"{len(recall_view):,}", border=True)
        metrics[1].metric("회수 사유 수", f"{recall_view['RTRVLPRVNS'].nunique():,}", border=True)
        metrics[2].metric("허가번호 확인 기록", f"{recall_view['LCNS_NO'].replace('', pd.NA).nunique():,}", border=True)
        chart_left, chart_right = st.columns(2)
        timeline = recall_view.dropna(subset=["CRET_DTM"]).assign(연도=lambda x: x["CRET_DTM"].dt.year.astype(str)).groupby("연도").size().reset_index(name="회수 기록 수")
        chart_left.plotly_chart(px.line(timeline, x="연도", y="회수 기록 수", markers=True, title="연도별 회수 기록"), width="stretch")
        reasons = recall_view["RTRVLPRVNS"].fillna("사유 미상").value_counts().head(15).rename_axis("회수 사유").reset_index(name="기록 수")
        top_reason_share = reasons.iloc[0]["기록 수"] / len(recall_view) * 100
        st.caption(f"현재 조건의 최다 회수 사유는 **{reasons.iloc[0]['회수 사유']}**로, 표시 기록의 {top_reason_share:.1f}%입니다. 이 비율은 공개 기록 내 구성비입니다.")
        chart_right.plotly_chart(px.bar(reasons, x="기록 수", y="회수 사유", orientation="h", title="회수 사유 분포"), width="stretch")
        products = recall_view["PRDLST_CD_NM"].fillna("품목 미상").value_counts().head(25).rename_axis("품목").reset_index(name="기록 수")
        st.plotly_chart(px.treemap(products, path=["품목"], values="기록 수", title="회수 품목 구성"), width="stretch")
        grade_method = recall_view.groupby([recall_view["RTRVL_GRDCD_NM"].fillna("등급 미상"), recall_view["RTRVLPLANDOC_RTRVLMTHD"].fillna("방법 미상")]).size().reset_index(name="기록 수")
        grade_method.columns = ["회수등급", "회수방법", "기록 수"]
        grade_method_pivot = grade_method.pivot(index="회수방법", columns="회수등급", values="기록 수").fillna(0)
        with st.container(border=True):
            st.subheader("회수등급·회수방법 매트릭스")
            st.caption("등급별로 기록된 회수방법의 구성을 비교합니다. 등급은 위험점수가 아니라 원자료의 회수 구분입니다.")
            st.plotly_chart(px.imshow(grade_method_pivot, aspect="auto", labels={"x":"회수등급", "y":"회수방법", "color":"기록 수"}, color_continuous_scale="Teal"), width="stretch")
        with st.expander("회수·판매중지 상세 기록", icon=":material/table_chart:"):
            recall_columns = ["CRET_DTM", "LCNS_NO", "BSSHNM", "PRDTNM", "PRDLST_CD_NM", "RTRVLPRVNS", "RTRVL_GRDCD_NM", "RTRVLPLANDOC_RTRVLMTHD", "ADDR"]
            show_dataframe(recall_view[recall_columns].sort_values("CRET_DTM", ascending=False).head(500), hide_index=True, width="stretch")
            st.download_button("현재 회수 분석 결과 다운로드", recall_view.to_csv(index=False).encode("utf-8-sig"), "foodsafe_recall_filtered.csv", "text/csv", icon=":material/download:")

elif page == "지역·지도 분석":
    st.title("지역·지도 분석", icon=":material/map:")
    render_page_guide(page)
    st.caption("HACCP 지정번호로 정확히 연결된 좌표와 지역 정보를 사용합니다. 좌표가 없는 인증은 지도에서 제외됩니다.")
    if haccp_geocoded.empty:
        st.info("HACCP–좌표 연결 데이터가 없습니다.")
    else:
        mapped = haccp_geocoded.copy()
        x_columns = [column for column in mapped.columns if "(x)" in column]
        y_columns = [column for column in mapped.columns if "(y)" in column]
        if not x_columns or not y_columns:
            st.warning("좌표 컬럼을 확인할 수 없습니다.")
        else:
            mapped["lon"] = pd.to_numeric(mapped[x_columns[0]], errors="coerce")
            mapped["lat"] = pd.to_numeric(mapped[y_columns[0]], errors="coerce")
            map_filter_a, map_filter_b, map_filter_c = st.columns(3)
            selected_area1 = map_filter_a.selectbox("시도", ["전체"] + sorted(mapped["area1"].dropna().unique().tolist()), key="map_area1")
            area_frame = mapped if selected_area1 == "전체" else mapped[mapped["area1"] == selected_area1]
            selected_area2 = map_filter_b.selectbox("시군구", ["전체"] + sorted(area_frame["area2"].dropna().unique().tolist()), key="map_area2")
            if selected_area2 != "전체":
                area_frame = area_frame[area_frame["area2"] == selected_area2]
            selected_map_industries = map_filter_c.multiselect("업종", sorted(area_frame["INDUTY_CD_NM"].dropna().unique()), key="map_industries")
            if selected_map_industries:
                area_frame = area_frame[area_frame["INDUTY_CD_NM"].isin(selected_map_industries)]
            coordinate_frame = area_frame.dropna(subset=["lat", "lon"])
            map_metrics = st.columns(3)
            map_metrics[0].metric("정확 연결 인증", f"{area_frame['appointno'].nunique():,}", border=True)
            map_metrics[1].metric("좌표 보유 인증", f"{coordinate_frame['appointno'].nunique():,}", border=True)
            map_metrics[2].metric("표시 시군구", f"{area_frame['area2'].nunique():,}", border=True)
            map_left, map_right = st.columns([3, 2])
            with map_left.container(border=True):
                st.subheader("HACCP 인증 위치")
                st.caption("좌표가 확인된 인증만 표시합니다.")
                st.map(coordinate_frame[["lat", "lon"]])
            with map_right.container(border=True):
                st.subheader("시군구별 인증 구성")
                st.caption("선택한 지역 안에서 시군구별 정확 연결 인증 수를 비교합니다.")
                area_counts = area_frame.groupby("area2")["appointno"].nunique().nlargest(20).reset_index(name="인증 수")
                st.plotly_chart(px.bar(area_counts, x="인증 수", y="area2", orientation="h", labels={"area2":"시군구"}), width="stretch")
            with st.expander("지도 표시 인증 명부", icon=":material/table_chart:"):
                show_dataframe(area_frame[["appointno", "BSSH_NM", "INDUTY_CD_NM", "businessitemNm", "area1", "area2", "source_type"]].head(1000), hide_index=True, width="stretch")
                st.download_button("현재 지역 결과 다운로드", area_frame.to_csv(index=False).encode("utf-8-sig"), "foodsafe_geocoded_filtered.csv", "text/csv", icon=":material/download:")

elif page == "행정처분 분석":
    st.title("행정처분 현황", icon=":material/gavel:")
    render_page_guide(page)
    if administrative.empty:
        st.info("원본 파일을 찾지 못했습니다.")
    else:
        admin_view = administrative.copy()
        admin_view["표시 시도"] = region_series(admin_view, ("ADDR", "SITE_ADDR"))
        admin_filter_a, admin_filter_b, admin_filter_c = st.columns(3)
        selected_admin_regions = admin_filter_a.multiselect("지역", sorted(admin_view["표시 시도"].dropna().unique()), key="admin_regions")
        selected_admin_types = admin_filter_b.multiselect("처분 유형", sorted(admin_view["DSPS_TYPECD_NM"].dropna().unique()), key="admin_types")
        selected_admin_industries = admin_filter_c.multiselect("업종", sorted(admin_view["INDUTY_CD_NM"].dropna().unique()), key="admin_industries")
        if selected_admin_regions:
            admin_view = admin_view[admin_view["표시 시도"].isin(selected_admin_regions)]
        if selected_admin_types:
            admin_view = admin_view[admin_view["DSPS_TYPECD_NM"].isin(selected_admin_types)]
        if selected_admin_industries:
            admin_view = admin_view[admin_view["INDUTY_CD_NM"].isin(selected_admin_industries)]
        if admin_view.empty:
            st.info("현재 선택 조건에 해당하는 행정처분 기록이 없습니다. 필터를 조정해 보세요.")
            st.stop()
        admin_metrics = st.columns(3)
        admin_metrics[0].metric("처분 기록", f"{len(admin_view):,}", border=True)
        admin_metrics[1].metric("고유 허가번호", f"{admin_view['LCNS_NO'].nunique():,}", border=True)
        admin_metrics[2].metric("처분기관", f"{admin_view['DSPS_INSTTCD_NM'].nunique():,}", border=True)
        c1, c2 = st.columns(2)
        by_type = admin_view["DSPS_TYPECD_NM"].fillna("미상").value_counts().head(15).rename_axis("처분 유형").reset_index(name="건수")
        top_admin_share = by_type.iloc[0]["건수"] / len(admin_view) * 100
        st.caption(f"현재 조건의 최다 처분 유형은 **{by_type.iloc[0]['처분 유형']}**으로, 표시 기록의 {top_admin_share:.1f}%입니다. 처분 기록 수는 업체 수와 다릅니다.")
        c1.plotly_chart(px.bar(by_type, x="건수", y="처분 유형", orientation="h", title="처분 유형별 건수"), width="stretch")
        by_industry = admin_view["INDUTY_CD_NM"].fillna("미상").value_counts().head(15).rename_axis("업종").reset_index(name="건수")
        c2.plotly_chart(px.bar(by_industry, x="건수", y="업종", orientation="h", title="업종별 처분 건수"), width="stretch")
        c3, c4 = st.columns(2)
        yearly = admin_view.dropna(subset=["DSPS_DCSNDT"]).assign(연도=lambda x: x["DSPS_DCSNDT"].dt.year.astype(str)).groupby("연도").size().reset_index(name="처분 건수")
        c3.plotly_chart(px.line(yearly, x="연도", y="처분 건수", markers=True, title="연도별 처분 추이"), width="stretch")
        tree = admin_view.groupby([admin_view["INDUTY_CD_NM"].fillna("업종 미상"), admin_view["DSPS_TYPECD_NM"].fillna("유형 미상")]).size().reset_index(name="건수")
        tree.columns = ["업종", "처분 유형", "건수"]
        c4.plotly_chart(px.treemap(tree, path=["업종", "처분 유형"], values="건수", title="업종·처분유형 구성"), width="stretch")
        heat_source = admin_view.dropna(subset=["DSPS_DCSNDT"]).assign(
            표시_시도=lambda x: region_series(x, ("ADDR", "SITE_ADDR")),
            연도=lambda x: x["DSPS_DCSNDT"].dt.year.astype(str),
        )
        heat = heat_source.groupby(["표시_시도", "연도"]).size().unstack(fill_value=0)
        if not heat.empty:
            st.plotly_chart(px.imshow(heat, aspect="auto", labels={"x":"연도", "y":"시도", "color":"처분 건수"}, title="시도·연도별 처분 집중도", color_continuous_scale="Teal"), width="stretch")
        law_type = admin_view.groupby([admin_view["LAWORD_CD_NM"].fillna("법령 미상"), admin_view["DSPS_TYPECD_NM"].fillna("유형 미상")]).size().reset_index(name="건수")
        law_type.columns = ["적용 법령", "처분 유형", "건수"]
        top_laws = law_type.groupby("적용 법령")["건수"].sum().nlargest(20).index
        law_matrix = law_type[law_type["적용 법령"].isin(top_laws)].pivot(index="적용 법령", columns="처분 유형", values="건수").fillna(0)
        duration_frame = admin_view.copy()
        duration_frame["처분 시작일"] = pd.to_datetime(duration_frame["DSPS_BGNDT"], errors="coerce", format="mixed")
        duration_frame["처분 종료일"] = pd.to_datetime(duration_frame["DSPS_ENDDT"], errors="coerce", format="mixed")
        duration_frame["처분 기간(일)"] = (duration_frame["처분 종료일"] - duration_frame["처분 시작일"]).dt.days
        duration_frame = duration_frame[duration_frame["처분 기간(일)"].ge(0)]
        admin_extra_left, admin_extra_right = st.columns(2)
        with admin_extra_left.container(border=True):
            st.subheader("적용 법령·처분유형 관계")
            st.caption("적용 법령과 처분유형이 함께 기록된 빈도를 보여줍니다.")
            st.plotly_chart(px.imshow(law_matrix, aspect="auto", labels={"x":"처분 유형", "y":"적용 법령", "color":"건수"}, color_continuous_scale="Teal"), width="stretch")
        with admin_extra_right.container(border=True):
            st.subheader("처분기간 분포")
            st.caption(f"시작일과 종료일을 모두 날짜로 확인할 수 있는 {len(duration_frame):,}건만 사용합니다.")
            st.plotly_chart(px.box(duration_frame, x="DSPS_TYPECD_NM", y="처분 기간(일)", labels={"DSPS_TYPECD_NM":"처분 유형"}), width="stretch")
        with st.expander("행정처분 상세 기록", icon=":material/table_chart:"):
            show_dataframe(admin_view[["LCNS_NO", "PRCSCITYPOINT_BSSHNM", "INDUTY_CD_NM", "DSPS_DCSNDT", "DSPS_TYPECD_NM", "LAWORD_CD_NM", "DSPSCN", "ADDR"]].sort_values("DSPS_DCSNDT", ascending=False).head(500), hide_index=True, width="stretch")
            st.download_button("현재 행정처분 결과 다운로드", admin_view.to_csv(index=False).encode("utf-8-sig"), "foodsafe_administrative_filtered.csv", "text/csv", icon=":material/download:")

elif page == "우선 확인 후보":
    st.title("우선 확인 후보", icon=":material/manage_search:")
    render_page_guide(page)
    st.caption("위험 예측이나 위험점수가 아닙니다. HACCP 인증 명부와 행정처분 이력이 허가번호로 연결되는 실제 관측 교집합입니다.")
    if prepared_candidates.empty:
        st.info("전처리 결과가 없습니다. `analysis/preprocess.py`를 먼저 실행하세요.")
    else:
        candidates = date_column(prepared_candidates, "DSPS_DCSNDT")
        with st.container(border=True):
            st.caption("업종 또는 HACCP 주소 기준 시도를 선택하면 아래 지표·차트·후보 목록이 함께 바뀝니다. 선택하지 않으면 전체를 표시합니다.")
            filter_left, filter_right = st.columns(2)
            selected_candidate_regions = filter_left.multiselect(
                "HACCP 주소 기준 시도", sorted(candidates["haccp_sido"].dropna().unique()), key="candidate_regions"
            )
            selected_candidate_industries = filter_right.multiselect(
                "업종", sorted(candidates["industry"].dropna().unique()), key="candidate_industries"
            )
        if selected_candidate_regions:
            candidates = candidates[candidates["haccp_sido"].isin(selected_candidate_regions)]
        if selected_candidate_industries:
            candidates = candidates[candidates["industry"].isin(selected_candidate_industries)]
        if candidates.empty:
            st.info("현재 선택 조건에 해당하는 후보가 없습니다. 필터를 조정해 보세요.")
            st.stop()
        c1, c2 = st.columns(2)
        c1.metric("허가번호로 연결된 행정처분 기록", f"{candidates['DSPSDTLS_SEQ'].nunique():,}")
        c2.metric("연결된 HACCP 업체", f"{candidates['LCNS_NO'].nunique():,}")
        st.warning("이 목록은 위험 예측이 아닙니다. 한 행은 실제 행정처분 1건이며, HACCP 업체 여부를 허가번호로 확인한 추가 검토 대상입니다.")
        chart_left, chart_right = st.columns(2)
        candidate_type = (
            candidates["DSPS_TYPECD_NM"].fillna("유형 미상").value_counts().head(12).rename_axis("처분 유형").reset_index(name="연결 기록 수")
        )
        chart_left.plotly_chart(
            px.bar(candidate_type, x="연결 기록 수", y="처분 유형", orientation="h", title="후보의 처분 유형 분포"),
            width="stretch",
        )
        candidate_year = (
            candidates.dropna(subset=["DSPS_DCSNDT"])
            .assign(연도=lambda x: x["DSPS_DCSNDT"].dt.year.astype(str))
            .groupby("연도")
            .size()
            .reset_index(name="연결 기록 수")
        )
        chart_right.plotly_chart(
            px.line(candidate_year, x="연도", y="연결 기록 수", markers=True, title="후보의 처분 결정 연도 분포"),
            width="stretch",
        )
        candidate_tree = (
            candidates.assign(
                업종=candidates["industry"].fillna("업종 미상"),
                처분유형=candidates["DSPS_TYPECD_NM"].fillna("유형 미상"),
            )
            .groupby(["업종", "처분유형"])
            .size()
            .reset_index(name="연결 기록 수")
        )
        st.plotly_chart(
            px.treemap(candidate_tree, path=["업종", "처분유형"], values="연결 기록 수", title="후보의 업종·처분유형 구성"),
            width="stretch",
        )
        entity_options = (
            candidates.sort_values("DSPS_DCSNDT", ascending=False)
            .drop_duplicates("LCNS_NO")
            .assign(
                후보표시=lambda x: x["company_name"].fillna("상호 미상")
                + " · 허가번호 "
                + x["LCNS_NO"].fillna("미상")
            )
        )
        selected_entity_label = st.selectbox(
            "상세 확인할 후보 업체",
            options=entity_options["후보표시"].tolist(),
            key="candidate_entity_detail",
        )
        selected_license = entity_options.loc[
            entity_options["후보표시"] == selected_entity_label, "LCNS_NO"
        ].iloc[0]
        entity_detail = candidates[candidates["LCNS_NO"] == selected_license].sort_values("DSPS_DCSNDT", ascending=False)
        entity_summary = entity_detail.iloc[0]
        certification_count = pd.to_numeric(entity_detail["haccp_certification_count"], errors="coerce").max()
        product_count = pd.to_numeric(entity_detail["product_type_count"], errors="coerce").max()
        with st.container(border=True):
            st.subheader("후보 상세 확인", icon=":material/manage_search:")
            st.info(
                f"선정 근거: HACCP 인증 명부와 행정처분 기록이 허가번호 {selected_license}로 실제 연결되었습니다. "
                "이는 위반 또는 위험의 확정 판단이 아니라 추가 확인을 위한 관측 결과입니다."
            )
            detail_left, detail_center, detail_right = st.columns(3)
            detail_left.metric("연결된 처분 기록", f"{entity_detail['DSPSDTLS_SEQ'].nunique():,}")
            detail_center.metric("HACCP 지정번호 수", f"{int(certification_count):,}" if pd.notna(certification_count) else "자료 없음")
            detail_right.metric("등록 품목 유형 수", f"{int(product_count):,}" if pd.notna(product_count) else "자료 없음")
            st.caption(
                f"업체: {entity_summary['company_name']} · 업종: {entity_summary['industry']} · "
                f"HACCP 주소 기준 시도: {entity_summary['haccp_sido']}"
            )
            linked_nonconforming = nonconforming[nonconforming.get("LCNS_NO", pd.Series(index=nonconforming.index, dtype=str)) == selected_license]
            linked_recalls = recalls[recalls.get("LCNS_NO", pd.Series(index=recalls.index, dtype=str)) == selected_license]
            with st.expander("Why? 후보 선정 근거", icon=":material/help:"):
                st.markdown("✓ **HACCP 인증 명부 확인**: 허가번호가 HACCP 명부에 존재합니다.")
                st.markdown(f"✓ **행정처분 데이터 연결**: 같은 허가번호의 처분 기록 {entity_detail['DSPSDTLS_SEQ'].nunique():,}건이 확인됩니다.")
                st.markdown(f"{'✓' if not linked_nonconforming.empty else '–'} **부적합 검사 직접 연결**: {'같은 허가번호 기록 ' + str(len(linked_nonconforming)) + '건' if not linked_nonconforming.empty else '직접 연결된 기록이 없습니다.'}")
                st.markdown(f"{'✓' if not linked_recalls.empty else '–'} **회수 직접 연결**: {'같은 허가번호 기록 ' + str(len(linked_recalls)) + '건' if not linked_recalls.empty else '직접 연결된 기록이 없습니다.'}")
                st.markdown("– **업체 연속성·재인허가**: 현재 결합 가능한 근거 데이터가 없어 판정하지 않습니다.")
                st.caption("위 항목은 공개 데이터의 연결 가능 여부를 설명하며, 안전·위험 판정이나 점수가 아닙니다.")
            detail_columns = ["DSPS_DCSNDT", "DSPS_TYPECD_NM", "DSPSCN", "VILTCN", "DSPS_INSTTCD_NM"]
            show_dataframe(entity_detail[detail_columns], hide_index=True, width="stretch")
        columns = ["LCNS_NO", "company_name", "industry", "haccp_sido", "haccp_certification_count", "product_type_count", "DSPS_DCSNDT", "DSPS_TYPECD_NM", "DSPSCN"]
        with st.container(border=True):
            st.subheader("후보 목록")
            show_dataframe(candidates[columns].sort_values("DSPS_DCSNDT", ascending=False).head(1000), hide_index=True, width="stretch")

elif page == "반복 사건 기록":
    st.title("반복 사건 기록", icon=":material/event_repeat:")
    render_page_guide(page)
    st.caption("같은 허가번호에 2회 이상 기록된 사건을 보여줍니다. 위험 판정이나 발생률이 아니라 공개데이터상 반복 기록입니다.")
    repeat_frames = []
    for label, frame, event_key in [
        ("행정처분", administrative, "DSPSDTLS_SEQ"),
        ("부적합 검사", nonconforming, "RTRVLDSUSE_SEQ"),
        ("회수", recalls, "RTRVLDSUSE_SEQ"),
    ]:
        counts = frame.dropna(subset=["LCNS_NO"]).groupby("LCNS_NO")[event_key].nunique().rename(label)
        repeat_frames.append(counts)
    repeat_matrix = pd.concat(repeat_frames, axis=1).fillna(0).astype(int)
    repeat_matrix = repeat_matrix[repeat_matrix.max(axis=1).ge(2)].reset_index()
    name_lookup = pd.concat([food[["LCNS_NO", "BSSH_NM"]], livestock[["LCNS_NO", "BSSH_NM"]]], ignore_index=True).drop_duplicates("LCNS_NO")
    repeat_matrix = repeat_matrix.merge(name_lookup, on="LCNS_NO", how="left")
    repeat_metrics = st.columns(3)
    repeat_metrics[0].metric("행정처분 2회 이상", f"{(repeat_matrix['행정처분'] >= 2).sum():,}", border=True)
    repeat_metrics[1].metric("부적합 검사 2회 이상", f"{(repeat_matrix['부적합 검사'] >= 2).sum():,}", border=True)
    repeat_metrics[2].metric("회수 2회 이상", f"{(repeat_matrix['회수'] >= 2).sum():,}", border=True)
    top_repeat = repeat_matrix.assign(전체=lambda x: x[["행정처분", "부적합 검사", "회수"]].sum(axis=1)).nlargest(50, "전체")
    matrix = top_repeat.set_index(top_repeat["BSSH_NM"].fillna(top_repeat["LCNS_NO"]))[["행정처분", "부적합 검사", "회수"]]
    with st.container(border=True):
        st.subheader("업체별 반복 사건 매트릭스")
        st.caption("색이 진할수록 해당 유형의 고유 사건 기록이 많습니다. 상위 50개 허가번호를 표시합니다.")
        st.plotly_chart(px.imshow(matrix, aspect="auto", labels={"x":"사건 유형", "y":"업체", "color":"기록 수"}, color_continuous_scale="Teal"), width="stretch")
    show_dataframe(repeat_matrix.sort_values(["행정처분", "부적합 검사", "회수"], ascending=False), hide_index=True, width="stretch")

elif page == "업체 통합 조회":
    st.title("업체 통합 조회", icon=":material/manage_search:")
    render_page_guide(page)
    st.caption("업체명 또는 허가번호로 찾은 뒤, 같은 허가번호로 정확히 연결되는 인증·검사·회수·행정처분 기록을 시간순으로 확인합니다.")
    registry_frames = []
    if not food.empty:
        registry_frames.append(food.assign(데이터구분="식품 HACCP"))
    if not livestock.empty:
        registry_frames.append(livestock.assign(데이터구분="축산물 HACCP"))
    if not registry_frames:
        st.info("조회할 HACCP 명부가 없습니다.")
    else:
        registry = pd.concat(registry_frames, ignore_index=True)
        query = st.text_input("업체명·허가번호 검색", placeholder="예: 상호 일부 2~4글자 또는 허가번호", key="company_search")
        with st.popover("검색 방법", icon=":material/help:"):
            st.markdown("- 등록 상호의 일부나 허가번호를 입력하세요.\n- 대표자명, 주소, 제품명은 이 검색창에서 찾지 않습니다.\n- 검색 결과가 많으면 업체 선택 목록에서 정확한 상호와 허가번호를 고르세요.")
        if not query.strip():
            st.info("업체명 일부 또는 허가번호를 입력하면, 정확한 식별키로 연결된 사건을 조회합니다.")
        else:
            query_text = query.strip()
            name_match = registry["BSSH_NM"].fillna("").str.contains(query_text, case=False, regex=False)
            licence_match = registry["LCNS_NO"].fillna("").astype(str).str.contains(query_text, case=False, regex=False)
            matched = registry[name_match | licence_match].copy()
            matched = matched.drop_duplicates(["데이터구분", "LCNS_NO"])
            if matched.empty:
                st.warning("일치하는 업체가 없습니다. 상호를 더 짧게 입력하거나 허가번호를 확인해 보세요.")
            else:
                matched["정확_허가번호"] = matched["LCNS_NO"].fillna("").astype(str).eq(query_text)
                matched = matched.sort_values(["정확_허가번호", "BSSH_NM"], ascending=[False, True])
                if len(matched) > 100:
                    st.warning(f"검색 결과가 {len(matched):,}건입니다. 정확한 상호나 허가번호로 범위를 줄여 주세요. 선택 목록에는 상위 100건만 표시합니다.")
                    matched = matched.head(100)
                else:
                    st.caption(f"일치하는 고유 허가번호 {len(matched):,}개를 찾았습니다.")
                matched["조회표시"] = (
                    matched["BSSH_NM"].fillna("상호 미상")
                    + " · "
                    + matched["데이터구분"]
                    + " · 허가번호 "
                    + matched["LCNS_NO"].fillna("미상")
                )
                selected_label = st.selectbox("조회할 업체", matched["조회표시"].tolist(), key="company_lookup")
                selected = matched.loc[matched["조회표시"] == selected_label].iloc[0]
                selected_license = selected["LCNS_NO"]
                haccp_rows = registry[registry["LCNS_NO"] == selected_license]
                linked_admin = administrative[administrative["LCNS_NO"] == selected_license]
                linked_inspections = nonconforming[nonconforming["LCNS_NO"] == selected_license]
                linked_recalls = recalls[recalls["LCNS_NO"] == selected_license]
                company_names = haccp_rows["BSSH_NM"].dropna().drop_duplicates().tolist()
                industries = haccp_rows["INDUTY_CD_NM"].dropna().drop_duplicates().tolist()
                addresses = haccp_rows["SITE_ADDR"].dropna().drop_duplicates().tolist()
                with st.container(border=True):
                    st.subheader("선택한 업체")
                    st.table(
                        {
                            "허가번호": str(selected_license),
                            "등록 상호": " / ".join(company_names[:3]) or "확인되지 않음",
                            "업종": " / ".join(industries[:3]) or "확인되지 않음",
                            "소재지": addresses[0] if addresses else "확인되지 않음",
                            "연결 기준": "LCNS_NO 정확 일치",
                        },
                        border="horizontal",
                        width="stretch",
                    )
                metrics = st.columns(4)
                metrics[0].metric("HACCP 지정번호", f"{haccp_rows['HACCP_APPN_NO'].nunique():,}", border=True)
                metrics[1].metric("부적합 검사 기록", f"{len(linked_inspections):,}", border=True)
                metrics[2].metric("회수 기록", f"{len(linked_recalls):,}", border=True)
                metrics[3].metric("행정처분 기록", f"{len(linked_admin):,}", border=True)
                connected_sources = ["HACCP 인증"]
                if not linked_inspections.empty:
                    connected_sources.append("부적합 검사")
                if not linked_recalls.empty:
                    connected_sources.append("회수·판매중지")
                if not linked_admin.empty:
                    connected_sources.append("행정처분")
                st.caption("현재 허가번호로 정확 연결된 데이터: " + " · ".join(connected_sources))
                timeline_rows = []
                for row in haccp_rows.drop_duplicates(["데이터구분", "HACCP_APPN_NO"]).itertuples(index=False):
                    certification_id = getattr(row, "HACCP_APPN_NO", "")
                    timeline_rows.append({"날짜": getattr(row, "HACCP_APPN_DT", None), "구분": "HACCP 지정", "사건·지정번호": certification_id, "내용": getattr(row, "INDUTY_CD_NM", "")})
                    timeline_rows.append({"날짜": getattr(row, "CRTFC_ENDDT", None), "구분": "인증 종료일", "사건·지정번호": certification_id, "내용": getattr(row, "INDUTY_CD_NM", "")})
                for row in linked_inspections.itertuples(index=False):
                    timeline_rows.append({"날짜": getattr(row, "CRET_DTM", None), "구분": "부적합 검사", "사건·지정번호": getattr(row, "RTRVLDSUSE_SEQ", ""), "내용": getattr(row, "TEST_ITMNM", "")})
                for row in linked_recalls.itertuples(index=False):
                    timeline_rows.append({"날짜": getattr(row, "CRET_DTM", None), "구분": "회수", "사건·지정번호": getattr(row, "RTRVLDSUSE_SEQ", ""), "내용": getattr(row, "RTRVLPRVNS", "")})
                for row in linked_admin.itertuples(index=False):
                    timeline_rows.append({"날짜": getattr(row, "DSPS_DCSNDT", None), "구분": "행정처분", "사건·지정번호": getattr(row, "DSPSDTLS_SEQ", ""), "내용": getattr(row, "DSPS_TYPECD_NM", "")})
                timeline = pd.DataFrame(timeline_rows)
                timeline["날짜"] = pd.to_datetime(timeline["날짜"], errors="coerce")
                timeline = timeline.dropna(subset=["날짜"])
                if not timeline.empty:
                    with st.container(border=True):
                        st.subheader("연결 기록 타임라인")
                        st.caption("동일 허가번호로 연결된 기록만 표시합니다. 빈 사건 유형은 안전을 뜻하지 않고 연결 기록이 없음을 뜻합니다.")
                        st.plotly_chart(px.scatter(timeline.sort_values("날짜"), x="날짜", y="구분", color="구분", hover_data=["사건·지정번호", "내용"], symbol="구분"), width="stretch")
                        st.download_button("업체 연결 타임라인 다운로드", timeline.sort_values("날짜").to_csv(index=False).encode("utf-8-sig"), f"foodsafe_company_{selected_license}_timeline.csv", "text/csv", icon=":material/download:")
                else:
                    st.info("날짜를 확인할 수 있는 연결 기록이 없습니다. 이는 안전을 의미하지 않습니다.")
                with st.expander("HACCP 인증 근거", icon=":material/verified:", expanded=True):
                    haccp_columns = [
                        column
                        for column in ["데이터구분", "BSSH_NM", "INDUTY_CD_NM", "PRDLST_NM", "HACCP_APPN_DT", "HACCP_APPN_NO", "SITE_ADDR"]
                        if column in haccp_rows.columns
                    ]
                    show_dataframe(haccp_rows[haccp_columns].head(500), hide_index=True, width="stretch")
                with st.expander("허가번호로 연결된 행정처분", icon=":material/gavel:"):
                    if linked_admin.empty:
                        st.caption("현재 행정처분 정제본에서 같은 허가번호의 기록을 찾지 못했습니다.")
                    else:
                        admin_columns = ["DSPS_DCSNDT", "DSPS_TYPECD_NM", "DSPSCN", "VILTCN", "DSPS_INSTTCD_NM"]
                        show_dataframe(linked_admin[admin_columns].sort_values("DSPS_DCSNDT", ascending=False), hide_index=True, width="stretch")
                if not linked_inspections.empty or not linked_recalls.empty:
                    with st.expander("허가번호로 연결된 검사·회수", icon=":material/fact_check:"):
                        if not linked_inspections.empty:
                            show_dataframe(linked_inspections[["CRET_DTM", "PRDTNM", "TEST_ITMNM", "STDR_STND", "TESTANALS_RSLT"]], hide_index=True, width="stretch")
                        if not linked_recalls.empty:
                            show_dataframe(linked_recalls[["CRET_DTM", "PRDTNM", "RTRVLPRVNS", "RTRVL_GRDCD_NM", "RTRVLPLANDOC_RTRVLMTHD"]], hide_index=True, width="stretch")
                else:
                    st.caption("현재 부적합 검사·회수 정제본에서 같은 허가번호의 기록을 찾지 못했습니다. 연결 불가능은 안전 판정이 아닙니다.")
                st.caption("부적합 검사·회수는 허가번호가 존재하고 정확히 일치하는 기록만 연결합니다.")

elif page == "맞춤 분석":
    st.title("맞춤 분석", icon=":material/tune:")
    st.caption("보고 싶은 주제와 그래프를 선택하면, 현재 데이터로 해석 가능한 시각화만 만들어 보여줍니다.")
    topic = st.segmented_control("분석 주제", ["HACCP 인증", "행정처분", "검사·회수"], default="HACCP 인증", key="custom_topic")
    if topic == "HACCP 인증":
        chart_choice = st.selectbox("그래프 선택", ["업종별 HACCP 지정번호", "지역별 HACCP 업체", "업종·품목 구성"], key="haccp_custom_chart")
        if food.empty:
            st.info("식품 HACCP 데이터가 없습니다.")
        elif chart_choice == "업종별 HACCP 지정번호":
            chart_data = food.groupby(food["INDUTY_CD_NM"].fillna("업종 미상"))["HACCP_APPN_NO"].nunique().sort_values(ascending=False).head(20).reset_index(name="HACCP 지정번호 수")
            chart_data.columns = ["업종", "HACCP 지정번호 수"]
            st.plotly_chart(px.bar(chart_data, x="HACCP 지정번호 수", y="업종", orientation="h", title=chart_choice), width="stretch")
        elif chart_choice == "지역별 HACCP 업체":
            chart_data = food.groupby(region_series(food, ("SITE_ADDR",)))["LCNS_NO"].nunique().sort_values(ascending=False).head(20).reset_index(name="고유 허가번호 수")
            chart_data.columns = ["시도", "고유 허가번호 수"]
            st.plotly_chart(px.bar(chart_data, x="시도", y="고유 허가번호 수", title=chart_choice), width="stretch")
        else:
            chart_data = food.assign(업종=food["INDUTY_CD_NM"].fillna("업종 미상"), 품목=food["PRDLST_NM"].fillna("품목 미상")).groupby(["업종", "품목"])["HACCP_APPN_NO"].nunique().reset_index(name="HACCP 지정번호 수")
            st.plotly_chart(px.treemap(chart_data, path=["업종", "품목"], values="HACCP 지정번호 수", title=chart_choice), width="stretch")
    elif topic == "행정처분":
        chart_choice = st.selectbox("그래프 선택", ["연도별 처분 추이", "처분 유형 분포", "지역·연도 집중도"], key="admin_custom_chart")
        if administrative.empty:
            st.info("행정처분 데이터가 없습니다.")
        elif chart_choice == "연도별 처분 추이":
            chart_data = administrative.dropna(subset=["DSPS_DCSNDT"]).assign(연도=lambda x: x["DSPS_DCSNDT"].dt.year.astype(str)).groupby("연도").size().reset_index(name="처분 기록 수")
            st.plotly_chart(px.line(chart_data, x="연도", y="처분 기록 수", markers=True, title=chart_choice), width="stretch")
        elif chart_choice == "처분 유형 분포":
            chart_data = administrative["DSPS_TYPECD_NM"].fillna("유형 미상").value_counts().head(20).rename_axis("처분 유형").reset_index(name="처분 기록 수")
            st.plotly_chart(px.bar(chart_data, x="처분 기록 수", y="처분 유형", orientation="h", title=chart_choice), width="stretch")
        else:
            chart_data = administrative.dropna(subset=["DSPS_DCSNDT"]).assign(표시_시도=lambda x: region_series(x, ("ADDR", "SITE_ADDR")), 연도=lambda x: x["DSPS_DCSNDT"].dt.year.astype(str)).groupby(["표시_시도", "연도"]).size().unstack(fill_value=0)
            st.plotly_chart(px.imshow(chart_data, aspect="auto", labels={"x": "연도", "y": "시도", "color": "처분 기록 수"}, color_continuous_scale="Teal", title=chart_choice), width="stretch")
    else:
        chart_choice = st.selectbox("그래프 선택", ["부적합 검사 항목", "회수 사유", "회수 품목 구성"], key="inspection_custom_chart")
        if chart_choice == "부적합 검사 항목" and not nonconforming.empty:
            chart_data = nonconforming["TEST_ITMNM"].fillna("검사 항목 미상").value_counts().head(20).rename_axis("검사 항목").reset_index(name="사례 수")
            st.plotly_chart(px.bar(chart_data, x="사례 수", y="검사 항목", orientation="h", title=chart_choice), width="stretch")
        elif chart_choice == "회수 사유" and not recalls.empty:
            chart_data = recalls["RTRVLPRVNS"].fillna("회수 사유 미상").value_counts().head(20).rename_axis("회수 사유").reset_index(name="사례 수")
            st.plotly_chart(px.bar(chart_data, x="사례 수", y="회수 사유", orientation="h", title=chart_choice), width="stretch")
        elif chart_choice == "회수 품목 구성" and not recalls.empty:
            chart_data = recalls["PRDLST_CD_NM"].fillna("품목 미상").value_counts().head(30).rename_axis("품목").reset_index(name="사례 수")
            st.plotly_chart(px.treemap(chart_data, path=["품목"], values="사례 수", title=chart_choice), width="stretch")
        else:
            st.info("선택한 그래프에 필요한 데이터가 없습니다.")
    st.caption("맞춤 분석은 탐색용 시각화입니다. 업체별 위험 예측이나 생산량 기반 비교는 현재 확보 데이터로 제공하지 않습니다.")

elif page == "사용 안내":
    st.title("FOODSAFE 사용 도우미", icon=":material/forum:")
    st.caption("현재 버전은 서비스 사용법과 데이터 해석 범위를 안내하는 규칙 기반 도우미입니다. 외부 AI 모델이나 개인 데이터를 사용하지 않습니다.")
    if "guide_messages" not in st.session_state:
        st.session_state.guide_messages = []
    if not st.session_state.guide_messages:
        st.info("아래 예시를 누르거나 궁금한 내용을 입력해 보세요.")
        suggestions = {
            "업체는 어떻게 검색하나요?": "업체는 어떻게 검색하나요?",
            "어떤 그래프를 볼 수 있나요?": "어떤 그래프를 볼 수 있나요?",
            "우선 확인 후보는 무슨 뜻인가요?": "우선 확인 후보는 무슨 뜻인가요?",
            "데이터 한계가 궁금해요": "데이터 한계가 궁금해요",
        }
        selected_suggestion = st.pills("예시 질문", list(suggestions), selection_mode="single", key="guide_suggestion")
        if selected_suggestion:
            st.session_state.guide_messages.append({"role": "user", "content": suggestions[selected_suggestion]})
            st.session_state.guide_messages.append({"role": "assistant", "content": guide_response(suggestions[selected_suggestion])})
            st.rerun()
    for message in st.session_state.guide_messages:
        with st.chat_message(message["role"], avatar=":material/support_agent:" if message["role"] == "assistant" else None):
            st.write(message["content"])
    prompt = st.chat_input("예: 업체는 어떻게 검색하나요?", key="guide_input")
    if prompt:
        st.session_state.guide_messages.append({"role": "user", "content": prompt})
        st.session_state.guide_messages.append({"role": "assistant", "content": guide_response(prompt)})
        st.rerun()
    if st.session_state.guide_messages and st.button("대화 초기화", icon=":material/restart_alt:"):
        st.session_state.guide_messages = []
        st.rerun()

elif page == "데이터 품질":
    st.title("데이터 품질", icon=":material/data_check:")
    st.caption("데이터의 규모·결측·중복·식별키·정확 연결 범위를 함께 공개합니다.")
    quality = pd.DataFrame(manifest.get("quality", [])) if manifest else pd.DataFrame()
    quality_metrics = st.columns(4)
    quality_metrics[0].metric("정제 데이터셋", f"{len(quality):,}", border=True)
    quality_metrics[1].metric("제거한 완전 중복", f"{pd.to_numeric(quality.get('removed_exact_duplicates', pd.Series(dtype=float)), errors='coerce').sum():,.0f}", border=True)
    coordinate_columns = [column for column in geocoded.columns if "(x)" in column or "(y)" in column]
    valid_coordinates = geocoded[coordinate_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).sum() if len(coordinate_columns) == 2 else 0
    quality_metrics[2].metric("유효 좌표", f"{valid_coordinates:,}", border=True)
    quality_metrics[3].metric("검사–회수 직접 연결", f"{inspection_recall_events['RTRVLDSUSE_SEQ'].nunique():,}", border=True)
    if not quality.empty:
        quality["키 보유율(%)"] = pd.to_numeric(quality["key_non_null_rows"], errors="coerce") / pd.to_numeric(quality["prepared_rows"], errors="coerce") * 100
        quality["주소 보유율(%)"] = pd.to_numeric(quality.get("address_non_null_rows"), errors="coerce") / pd.to_numeric(quality["prepared_rows"], errors="coerce") * 100
        completeness = quality.melt(id_vars="dataset", value_vars=["키 보유율(%)", "주소 보유율(%)"], var_name="품질 지표", value_name="보유율(%)")
        quality_left, quality_right = st.columns(2)
        with quality_left.container(border=True):
            st.subheader("데이터셋 규모")
            st.caption("완전 중복을 제거한 이후의 정제 행 수입니다.")
            st.plotly_chart(px.bar(quality, x="prepared_rows", y="dataset", orientation="h", labels={"prepared_rows":"정제 행 수", "dataset":"데이터셋"}), width="stretch")
        with quality_right.container(border=True):
            st.subheader("핵심 필드 보유율")
            st.caption("식별키와 주소가 실제로 채워진 행의 비율입니다.")
            st.plotly_chart(px.bar(completeness, x="보유율(%)", y="dataset", color="품질 지표", barmode="group", orientation="h", range_x=[0, 100]), width="stretch")
    if not linkage_audit.empty:
        audit = linkage_audit.copy()
        for column in ["prepared_rows", "licence_present_rows", "rows_exactly_linked_to_any_haccp"]:
            audit[column] = pd.to_numeric(audit[column], errors="coerce")
        audit["식별키 보유율(%)"] = audit["licence_present_rows"] / audit["prepared_rows"] * 100
        audit["HACCP 정확 연결률(%)"] = audit["rows_exactly_linked_to_any_haccp"] / audit["prepared_rows"] * 100
        audit_long = audit.melt(id_vars="dataset", value_vars=["식별키 보유율(%)", "HACCP 정확 연결률(%)"], var_name="단계", value_name="비율(%)")
        with st.container(border=True):
            st.subheader("식별키 보유와 HACCP 정확 연결")
            st.caption("낮은 연결률을 숨기지 않고 전체 기록을 분모로 표시합니다.")
            st.plotly_chart(px.bar(audit_long, x="dataset", y="비율(%)", color="단계", barmode="group", range_y=[0, 100]), width="stretch")
    show_dataframe(quality, hide_index=True, width="stretch")

elif page == "데이터·분석 방법":
    st.title("데이터·분석 방법", icon=":material/menu_book:")
    source_col, linkage_col = st.columns(2)
    with source_col:
        st.subheader("데이터 소스")
        st.markdown("- HACCP 인증\n- 행정처분\n- 부적합 검사\n- 회수·판매중지\n- 좌표 확인 HACCP 명부")
    with linkage_col:
        st.subheader("결합 기준")
        st.markdown("- HACCP ↔ 행정처분: `LCNS_NO`\n- 검사 ↔ 회수: `RTRVLDSUSE_SEQ`\n- HACCP ↔ 좌표: `HACCP_APPN_NO`\n- 주소 → 시도·시군구\n- 업체명은 자동 결합 키로 사용하지 않음")
    st.subheader("분석 방법")
    st.markdown("기술통계, 시계열, 지역·품목 분포, 실제 식별키 기반 데이터 연결을 사용합니다.")
    st.subheader("해석 한계")
    st.markdown("""
    - 행정처분은 실제 관측 이력이며 모든 처분을 식품안전 부적합과 동일시하지 않습니다.
    - 검사 부적합·회수 데이터는 업체 식별키 결측이 있어 업체별 위험 예측 라벨로 사용하지 않습니다.
    - 업체별 미래 부적합 예측, 임의 위험점수, 특정 업체의 위험 판정은 제공하지 않습니다.
    - CCP와 특정 업체의 직접 연결, 개선·후속관리 효과 분석은 현재 데이터로 제공하지 않습니다.
    - 우선 확인 후보는 위험 사실의 확정이 아니라 관리자의 추가 확인을 돕는 탐색 결과입니다.
    """)
    st.subheader("현재 보유 여부와 구현 범위")
    availability = pd.DataFrame([
        {"데이터": "HACCP 인증", "현재 보유": "O", "활용 수준": "즉시 활용", "대시보드 활용": "인증 분포·생애주기·업체 조회"},
        {"데이터": "행정처분", "현재 보유": "O", "활용 수준": "부분 연결", "대시보드 활용": "시간·지역·유형·법령 분석"},
        {"데이터": "부적합 검사", "현재 보유": "O", "활용 수준": "부분 연결", "대시보드 활용": "검사항목·품목·기관 분석"},
        {"데이터": "회수·판매중지", "현재 보유": "O", "활용 수준": "부분 연결", "대시보드 활용": "사유·등급·방법·검사 연결"},
        {"데이터": "개선조치 결과", "현재 보유": "X", "활용 수준": "구현 불가", "대시보드 활용": "조치일·완료일·결과 필요"},
        {"데이터": "후속점검", "현재 보유": "X", "활용 수준": "구현 불가", "대시보드 활용": "점검일·점검결과·재검사 결과 필요"},
        {"데이터": "업체별 CCP·공정", "현재 보유": "X", "활용 수준": "구현 불가", "대시보드 활용": "업체 식별키·CCP·위해요소 필요"},
        {"데이터": "생산량", "현재 보유": "X", "활용 수준": "구현 불가", "대시보드 활용": "업체·품목·기간별 생산량 필요"},
    ])
    show_dataframe(availability, hide_index=True, width="stretch")
    st.caption("※ X로 표시된 항목은 현재 프로젝트 데이터인 것처럼 생성하지 않으며, 화면에서도 분석 결과로 제시하지 않습니다.")
    if manifest:
        st.subheader("현재 전처리 품질 요약")
        show_dataframe(pd.DataFrame(manifest.get("quality", [])), hide_index=True, width="stretch")
        st.caption("행 수 차이는 원본에서 제거한 완전 중복 행이며, 원본 파일 자체는 변경하지 않습니다.")
    st.subheader("식별키 연결 가능성")
    st.caption("아래 표는 자동 상호·주소 매칭 없이 `LCNS_NO`가 정확히 일치한 기록만 계산합니다. 연결되지 않은 기록은 업체 단위 분석에 사용하지 않습니다.")
    if linkage_audit.empty:
        st.info("연결성 감사표가 없습니다. `analysis/preprocess.py`를 실행하면 생성됩니다.")
    else:
        show_dataframe(
            linkage_audit,
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "연결성 감사표 다운로드",
            linkage_audit.to_csv(index=False).encode("utf-8-sig"),
            "foodsafe_linkage_audit.csv",
            "text/csv",
            icon=":material/download:",
        )
