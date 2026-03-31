from __future__ import annotations

import unittest

from remote_infer.utils import (
    DecodedImageSlice,
    build_medgemma_image_report_messages,
    build_medgemma_text_messages,
    prepare_medgemma_multimodal_inputs,
    prepare_medgemma_text_inputs,
)


class _FakeChatTemplateIO:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
        self.calls.append((messages, kwargs))
        return {"input_ids": "tokenized"}


class _FallbackChatTemplateIO:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
        self.calls.append((messages, kwargs))
        if kwargs.get("tokenize") is True:
            raise RuntimeError("tokenized template path unavailable")
        return "<chat prompt>"

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
        self.calls.append((prompt, kwargs))
        return {"input_ids": prompt}


class GenerationFormattingTests(unittest.TestCase):
    def test_builds_single_user_turn_message(self) -> None:
        prompt = "Draft a concise English radiology report."

        messages = build_medgemma_text_messages(prompt)

        self.assertEqual(
            messages,
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

    def test_prepare_inputs_uses_chat_template_with_generation_prompt(self) -> None:
        text_io = _FakeChatTemplateIO()
        prompt = "Caller supplied full prompt"

        inputs = prepare_medgemma_text_inputs(text_io, prompt)

        self.assertEqual(inputs, {"input_ids": "tokenized"})
        self.assertEqual(len(text_io.calls), 1)
        messages, kwargs = text_io.calls[0]
        self.assertEqual(messages, build_medgemma_text_messages(prompt))
        self.assertEqual(
            kwargs,
            {
                "add_generation_prompt": True,
                "continue_final_message": False,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            },
        )

    def test_prepare_inputs_falls_back_to_string_template_when_needed(self) -> None:
        text_io = _FallbackChatTemplateIO()

        inputs = prepare_medgemma_text_inputs(text_io, "Caller supplied full prompt")

        self.assertEqual(inputs, {"input_ids": "<chat prompt>"})
        self.assertEqual(len(text_io.calls), 3)
        self.assertEqual(text_io.calls[0][1]["tokenize"], True)
        self.assertEqual(text_io.calls[0][1]["continue_final_message"], False)
        self.assertEqual(text_io.calls[1][1]["tokenize"], False)
        self.assertEqual(text_io.calls[1][1]["continue_final_message"], False)
        self.assertEqual(text_io.calls[2][0], "<chat prompt>")

    def test_builds_notebook_style_multimodal_message(self) -> None:
        slices = [
            DecodedImageSlice(slice_index=119, relative_position="n-1", anchor_label="A01 n-1", image="img-1"),
            DecodedImageSlice(slice_index=120, relative_position="n", anchor_label="A01 n", image="img-2"),
            DecodedImageSlice(slice_index=121, relative_position="n+1", anchor_label="A01 n+1", image="img-3"),
        ]

        messages = build_medgemma_image_report_messages(
            instruction="Instruction: Review the contiguous CT slices.",
            slices=slices,
            query="Draft a concise radiology-style response.",
        )

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Instruction: Review the contiguous CT slices."},
                        {"type": "image", "image": "img-1"},
                        {"type": "text", "text": "SLICE A01 n-1 (index 119)"},
                        {"type": "image", "image": "img-2"},
                        {"type": "text", "text": "SLICE A01 n (index 120)"},
                        {"type": "image", "image": "img-3"},
                        {"type": "text", "text": "SLICE A01 n+1 (index 121)"},
                        {"type": "text", "text": "Draft a concise radiology-style response."},
                    ],
                }
            ],
        )

    def test_prepare_multimodal_inputs_uses_notebook_template_kwargs(self) -> None:
        processor = _FakeChatTemplateIO()
        messages = build_medgemma_image_report_messages(
            instruction="Instruction: Review the slices in order.",
            slices=[DecodedImageSlice(slice_index=120, relative_position="n", image="img-1")],
            query="Draft a concise radiology-style response.",
        )

        inputs = prepare_medgemma_multimodal_inputs(processor, messages)

        self.assertEqual(inputs, {"input_ids": "tokenized"})
        self.assertEqual(len(processor.calls), 1)
        _, kwargs = processor.calls[0]
        self.assertEqual(
            kwargs,
            {
                "add_generation_prompt": True,
                "continue_final_message": False,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            },
        )


if __name__ == "__main__":
    unittest.main()
