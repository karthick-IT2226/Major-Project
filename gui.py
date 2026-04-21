import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
from stable_baselines3 import DQN
from mental_env import MentalHealthEnv

# ---------------- LOAD MODEL ----------------
MODEL_PATH = r"dqn_mental_health finall.zip"

env = MentalHealthEnv()
model = DQN.load(MODEL_PATH, env=env)

# ---------------- GUI SETUP ----------------
root = tk.Tk()
root.title("Mental Health Risk Prediction System")
root.geometry("760x720")   # FIXED HEIGHT
root.configure(bg="#f5f6f7")

style = ttk.Style()
style.theme_use("default")

style.configure("TLabel", font=("Segoe UI", 10), background="#f5f6f7")
style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
style.configure("TButton", font=("Segoe UI", 10))
style.configure("Horizontal.TProgressbar", thickness=16)

# ---------------- HEADER ----------------
ttk.Label(
    root,
    text="Mental Health Risk Prediction",
    style="Header.TLabel"
).pack(pady=10)

ttk.Label(
    root,
    text="Deep Q-Network Based Decision Support System",
    foreground="gray"
).pack(pady=2)

# ---------------- INPUT FRAME ----------------
input_frame = ttk.LabelFrame(root, text="Behavioral Questions")
input_frame.pack(fill="x", padx=20, pady=15)

questions = [
    "How much time do you spend on social media in a day?",
    "How often do you feel distracted while using social media?",
    "How restless or irritable do you feel when not using social media?",
    "How frequently do you feel anxious or worried during the day?",
    "How often do you experience irregular or disturbed sleep?"
]

sliders = []

for i, question in enumerate(questions):
    ttk.Label(
        input_frame,
        text=question,
        wraplength=360,
        justify="left"
    ).grid(row=i, column=0, sticky="w", padx=10, pady=10)

    scale = ttk.Scale(
        input_frame,
        from_=0,
        to=1,
        orient="horizontal",
        length=260
    )
    scale.set(0.30)
    scale.grid(row=i, column=1, padx=10)

    value_label = ttk.Label(input_frame, text="0.30")
    value_label.grid(row=i, column=2, padx=10)

    def update_value(val, lbl=value_label):
        lbl.config(text=f"{float(val):.2f}")

    scale.config(command=update_value)
    sliders.append(scale)

# ---------------- OUTPUT FRAME ----------------
output_frame = ttk.LabelFrame(root, text="Prediction Output")
output_frame.pack(fill="x", padx=20, pady=15)

result_label = ttk.Label(
    output_frame,
    text="Risk Level: —",
    font=("Segoe UI", 12, "bold")
)
result_label.pack(pady=10)

confidence_bar = ttk.Progressbar(
    output_frame,
    orient="horizontal",
    length=460,
    mode="determinate"
)
confidence_bar.pack(pady=5)

confidence_text = ttk.Label(output_frame, text="Confidence Score: —")
confidence_text.pack(pady=5)

# Risk color mapping
risk_map = {
    0: ("Good", "#2ecc71"),
    1: ("Moderate", "#f1c40f"),
    2: ("Worsening", "#e67e22"),
    3: ("Severe", "#e74c3c")
}

# ---------------- PREDICTION FUNCTION ----------------
def predict():
    state = np.array([s.get() for s in sliders], dtype=np.float32)

    action, _ = model.predict(state, deterministic=True)

    with torch.no_grad():
        q_vals = model.q_net(
            torch.tensor(state).unsqueeze(0)
        ).cpu().numpy()[0]

    exp_q = np.exp(q_vals - np.max(q_vals))
    probs = exp_q / exp_q.sum()
    confidence = probs[int(action)] * 100

    label, color = risk_map[int(action)]

    result_label.config(
        text=f"Risk Level: {label}",
        foreground=color
    )

    confidence_bar["value"] = confidence
    confidence_text.config(text=f"Confidence Score: {confidence:.2f}%")

# ---------------- BUTTON (FIXED POSITION) ----------------
ttk.Button(
    output_frame,
    text="Predict Mental Health Risk",
    command=predict
).pack(pady=15)

# ---------------- START GUI ----------------
root.mainloop()
