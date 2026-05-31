import gradio as gr
import torch, json
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#The file path of "class_names.json" and "treatment_map.json" should be used here.
with open("/kaggle/working/class_names.json")   as f: classes       = json.load(f)
with open("/kaggle/working/treatment_map.json") as f: treatment_map = json.load(f)


inf_model = efficientnet_b4(weights=None)
in_features = inf_model.classifier[1].in_features
inf_model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 512),
    nn.GELU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, len(classes))
)

#The file path of the saved best model should be used here
inf_model.load_state_dict(torch.load("/kaggle/working/best_model.pth", map_location=DEVICE))
inf_model = inf_model.to(DEVICE)
inf_model.eval()

eval_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(image):
    if image is None:
        return "⚠️ Please upload a leaf image.", "", {}

    img_t = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(inf_model(img_t), dim=-1)[0]

    top5_p, top5_i = probs.topk(5)
    top5       = {classes[i]: round(p.item(), 4) for i, p in zip(top5_i, top5_p)}
    pred_cls   = classes[top5_i[0].item()]
    confidence = top5_p[0].item() * 100
    treatment  = treatment_map.get(pred_cls, "⚠️ Consult an agricultural expert.")

    result  = f"### 🌿 `{pred_cls.replace('_', ' ')}`\n\n"
    result += f"**Confidence:** {confidence:.1f}%\n\n---\n\n"
    result += f"**Treatment:**\n\n{treatment}"
    return result, pred_cls, top5

with gr.Blocks(title="🌿 Crop Disease Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 EfficientNet-B4 Crop Disease Detector\nUpload a leaf image to detect disease and get treatment advice.")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="📷 Leaf Image")
            btn    = gr.Button("🔍 Detect Disease", variant="primary", size="lg")
        with gr.Column():
            out_md    = gr.Markdown()
            out_label = gr.Textbox(label="Predicted Class", interactive=False)
            out_top5  = gr.Label(label="Top 5 Predictions", num_top_classes=5)

    btn.click(predict, inputs=[img_in], outputs=[out_md, out_label, out_top5])
    gr.Markdown("> Model: EfficientNet-B4 fine-tuned on PlantVillage (38 classes, ~54K images).")

demo.launch(share=True, debug=True)
