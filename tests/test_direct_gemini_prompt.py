from rag.direct_gemini import build_direct_baseline_prompt


def test_build_direct_baseline_prompt_contains_required_instructions() -> None:
    prompt = build_direct_baseline_prompt("การถอนวิชาเรียนทำยังไง")
    assert "ตอบเป็นภาษาไทยเท่านั้น" in prompt
    assert "ตอบให้กระชับและตรงคำถาม" in prompt
    assert "ห้ามใส่แหล่งอ้างอิง" in prompt
    assert "ถ้าไม่แน่ใจ" in prompt
    assert "การถอนวิชาเรียนทำยังไง" in prompt


def test_build_direct_baseline_prompt_includes_context_when_provided() -> None:
    prompt = build_direct_baseline_prompt(
        "ลงทะเบียนเมื่อไหร่",
        context_chunks=[
            "คำถาม: ระเบียบการลงทะเบียน คำตอบ: ลงทะเบียนตามปฏิทินการศึกษา",
            "คำถาม: การเพิ่มวิชา คำตอบ: ภายใน 3 สัปดาห์",
        ],
    )
    assert "CONTEXT:" in prompt
    assert "คำถาม: ระเบียบการลงทะเบียน" in prompt
    assert "คำถาม: การเพิ่มวิชา" in prompt
