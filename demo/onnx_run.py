import onnxruntime

class onnx_inferencer:

    def __init__(self, model_path) -> None:
        self.onnx_model_sess = onnxruntime.InferenceSession(model_path)
        self.output_names = []
        self.input_names = []
        print(model_path)
        for i in range(len(self.onnx_model_sess.get_inputs())):
            self.input_names.append(self.onnx_model_sess.get_inputs()[i].name)
            print("    input:", i,
                  self.onnx_model_sess.get_inputs()[i].name,self.onnx_model_sess.get_inputs()[i].type,
                  self.onnx_model_sess.get_inputs()[i].shape)

        for i in range(len(self.onnx_model_sess.get_outputs())):
            self.output_names.append(
                self.onnx_model_sess.get_outputs()[i].name)
            print("    output:", i,
                  self.onnx_model_sess.get_outputs()[i].name,self.onnx_model_sess.get_outputs()[i].type,
                  self.onnx_model_sess.get_outputs()[i].shape)
        print("")

    def get_input_count(self):
        return len(self.input_names)

    def get_input_shape(self, idx: int):
        return self.onnx_model_sess.get_inputs()[idx].shape

    def get_input_names(self):
        return self.input_names

    def get_output_count(self):
        return len(self.output_names)

    def get_output_shape(self, idx: int):
        return self.onnx_model_sess.get_outputs()[idx].shape

    def get_output_names(self):
        return self.output_names

    def inference(self, tensor):
        return self.onnx_model_sess.run(
            self.output_names, input_feed={self.input_names[0]: tensor})

    def inference_multi_input(self, tensors: list):
        inputs = dict()
        for idx, tensor in enumerate(tensors):
            inputs[self.input_names[idx]] = tensor
        return self.onnx_model_sess.run(self.output_names, input_feed=inputs)

backbone = onnx_inferencer("weights/backbone.onnx")
bert = onnx_inferencer("weights/bert.onnx")
transformer = onnx_inferencer("weights/transformer.onnx")

from groundingdino.util.inference import load_image, annotate
from groundingdino.util import get_tokenlizer
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from groundingdino.util.utils import get_phrases_from_posmap
import torch
import cv2
import numpy as np

tokenizer = get_tokenlizer.get_tokenlizer("bert-base-uncased")
specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

IMAGE_PATH = ".asset/cat_dog.jpeg"
TEXT_PROMPT = "chair . person . dog ."
image_source, image = load_image(IMAGE_PATH)

srcs = backbone.inference(image.unsqueeze(0).numpy())

tokenized = tokenizer(TEXT_PROMPT, padding="longest", return_tensors="pt")

(
    text_self_attention_masks,
    position_ids,
    cate_to_token_mask_list,
) = generate_masks_with_special_tokens_and_transfer_map(tokenized, specical_tokens, tokenizer)

tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
tokenized_for_encoder["attention_mask"] = text_self_attention_masks
tokenized_for_encoder["position_ids"] = position_ids

# print(tokenized_for_encoder)

encoded_text = bert.inference_multi_input([tokenized_for_encoder["input_ids"].numpy(), tokenized_for_encoder["attention_mask"].numpy(), tokenized_for_encoder["position_ids"].numpy()])
# print(encoded_text.shape)
prediction_logits, prediction_boxes = transformer.inference_multi_input(
            [*srcs, encoded_text[0], position_ids.numpy(), text_self_attention_masks.numpy()]
        )

# print(logits.shape)

prediction_logits = prediction_logits[0][0]  # prediction_logits.shape = (nq, 256)
prediction_boxes = prediction_boxes[0][0]  # prediction_boxes.shape = (nq, 4)
print(prediction_logits.shape,prediction_boxes.shape)
mask = np.max(prediction_logits,axis=1) > BOX_TRESHOLD
# print(mask)
logits = prediction_logits[mask]  # logits.shape = (n, 256)
boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

print(logits.shape,boxes.shape)

phrases = [
            get_phrases_from_posmap(torch.Tensor(logit > TEXT_TRESHOLD), tokenized, tokenizer).replace('.', '')
            for logit
            in logits[0]
        ]
logits = logits.max(axis=1)
print(logits)
annotated_frame = annotate(image_source=image_source, boxes=torch.Tensor(boxes), logits=torch.Tensor(logits), phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)