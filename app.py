import streamlit as st
import pandas as pd
from pathlib import Path

from service import analyze_drawing
from review import chat_about_drawing


# 페이지 설정: 화면을 넓게 사용
st.set_page_config(page_title="수변전 KEC/KS 피드백", layout="wide")

# -----------------------------
# 0. 세션 상태 초기화
# -----------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None

if "drawing_id" not in st.session_state:
    st.session_state["drawing_id"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "uploaded_image_path" not in st.session_state:
    st.session_state["uploaded_image_path"] = None

# -----------------------------
# 1. 페이지 상단 제목 영역
# -----------------------------
st.title("수변전 설비 도면 피드백")
st.caption("이미지(JPG/PNG) 1장 ")

# -----------------------------
# 2. 왼쪽 사이드바 (리모컨 역할)
# -----------------------------
with st.sidebar:
    st.header("① 도면 업로드 & 분석")

    uploaded_file = st.file_uploader(
        "도면 이미지 파일을 올려주세요",
        type=["png", "jpg", "jpeg"],
        help="수변전 설비 단선도 이미지 1장을 올리세요.",
    )

    run_button = st.button("분석 실행", use_container_width=True)

    if run_button:
        if uploaded_file is None:
            st.warning("먼저 도면 이미지를 업로드하세요.")
        else:
            # 업로드한 파일을 로컬에 저장 (임시 폴더)
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)

            img_path = uploads_dir / uploaded_file.name
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            st.session_state["uploaded_image_path"] = str(img_path)

            # 도면 분석 실행
            try:
                with st.spinner("YOLO + OCR + 규정 검사를 수행 중입니다..."):
                    result = analyze_drawing(str(img_path))

                # 분석 결과와 도면 ID 저장
                st.session_state["result"] = result
                st.session_state["drawing_id"] = f"DEMO-{img_path.stem}"
                st.success("분석이 완료되었습니다. 오른쪽 화면에서 결과를 확인하세요.")

            except Exception as e:
                st.session_state["result"] = None
                st.session_state["drawing_id"] = None
                st.error(f"분석 중 오류가 발생했습니다: {e}")

    st.markdown("---")
    st.subheader("도움말")
    st.markdown(
        """
        1. 도면 이미지를 업로드합니다.  
        2. **분석 실행** 버튼을 누릅니다.  
        3. 오른쪽에서 위반 요약, 상세 목록, 채팅을 확인합니다.  
        """
    )

# -----------------------------
# 3. 메인 화면 레이아웃 (왼쪽: 도면/요약, 오른쪽: 상세/채팅)
# -----------------------------
result = st.session_state.get("result")
drawing_id = st.session_state.get("drawing_id")

if result is None:
    st.info("왼쪽에서 도면을 업로드하고 **분석 실행** 버튼을 눌러주세요.")
else:
    # 화면을 두 컬럼으로 나눔
    col_left, col_right = st.columns([2, 3])

    # -------------------------
    # 3-1. 왼쪽: 도면 이미지 + 요약 카드
    # -------------------------
    with col_left:
        st.subheader("② 도면 미리보기")

        img_path = st.session_state.get("uploaded_image_path")
        if img_path is not None:
            st.image(img_path, caption="업로드한 도면", use_column_width=True)
        else:
            st.write("도면 이미지를 찾을 수 없습니다.")

        st.markdown("---")
        st.subheader("③ 위반 요약")

        summary = result.get("summary", {}) if isinstance(result, dict) else {}

        total = summary.get("total_violations", 0)
        high = summary.get("high", 0)
        medium = summary.get("medium", 0)
        low = summary.get("low", 0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 위반 수", total)
        m2.metric("HIGH", high)
        m3.metric("MEDIUM", medium)
        m4.metric("LOW", low)

    # -------------------------
    # 3-2. 오른쪽: 위반 상세 + 채팅 UI
    # -------------------------
    with col_right:
        st.subheader("④ 위반 상세 목록")

        violations = result.get("violations", []) if isinstance(result, dict) else []

        if violations:
            # violations는 dict 리스트라고 가정
            df = pd.DataFrame(violations)
            st.dataframe(df, use_container_width=True)
        else:
            st.write("위반 정보가 없습니다. (규정 위반이 없거나, 분석에 실패했을 수 있습니다.)")

        st.markdown("---")
        st.subheader("⑤ 도면/규정 관련 질문하기")

        # 지금까지의 대화 기록 다시 출력
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 사용자 입력
        user_q = st.chat_input("도면이나 KEC/KS 규정에 대해 궁금한 점을 물어보세요.")

        if user_q:
            if drawing_id is None:
                st.info("먼저 도면을 분석해 주세요.")
            else:
                # user 말풍선
                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_q}
                )
                with st.chat_message("user"):
                    st.markdown(user_q)

                # assistant 답변 말풍선
                with st.chat_message("assistant"):
                    try:
                        answer = chat_about_drawing(user_q, drawing_id)
                    except Exception as e:
                        answer = f"답변 중 오류가 발생했습니다: {e}"
                    st.markdown(answer)

                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer}
                )
