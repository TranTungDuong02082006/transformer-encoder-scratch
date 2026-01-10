import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
COLORS = {"train": "#2ecc71", "val": "#e74c3c", "lr": "#3498db", "ppl": "#9b59b6"}

LOG_DIR = "checkpoints"
OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_training_loss(df_train):
    """Vẽ biểu đồ Training Loss theo từng bước (Smoothing cho mượt)."""
    plt.figure(figsize=(10, 6))
    
    # Vẽ dữ liệu gốc (mờ)
    sns.lineplot(data=df_train, x="global_step", y="train_loss", 
                 alpha=0.3, color=COLORS["train"], label="Raw Loss")
    
    # Vẽ dữ liệu đã làm mượt (Rolling average) cho dễ nhìn xu hướng
    # window=50 nghĩa là trung bình 50 bước liên tiếp
    df_train["smoothed_loss"] = df_train["train_loss"].rolling(window=50).mean()
    sns.lineplot(data=df_train, x="global_step", y="smoothed_loss", 
                 linewidth=2, color=COLORS["train"], label="Smoothed Loss (MA-50)")
    
    plt.title("Training Loss Convergence", fontweight="bold")
    plt.xlabel("Global Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=300)
    print(f"Saved: {OUTPUT_DIR}/training_loss.png")

def plot_learning_rate(df_train):
    """Vẽ biểu đồ Learning Rate Schedule."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_train, x="global_step", y="learning_rate", 
                 color=COLORS["lr"], linewidth=2)
    
    plt.title("Learning Rate Schedule with Warmup", fontweight="bold")
    plt.xlabel("Global Training Steps")
    plt.ylabel("Learning Rate")
    plt.grid(True, which="minor", linestyle="--", alpha=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lr_schedule.png"), dpi=300)
    print(f"Saved: {OUTPUT_DIR}/lr_schedule.png")

def plot_validation_metrics(df_val):
    """Vẽ biểu đồ Validation Loss và PPL trên cùng một hình (2 trục Y)."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Trục Y bên trái: Validation Loss
    sns.lineplot(data=df_val, x="epoch", y="val_loss", 
                 ax=ax1, color=COLORS["val"], marker="o", linewidth=2, label="Val Loss")
    ax1.set_ylabel("Validation Loss", color=COLORS["val"], fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=COLORS["val"])
    ax1.set_xlabel("Epochs", fontweight="bold")
    ax1.set_xticks(df_val["epoch"].unique()) # Đảm bảo trục X chỉ hiện số nguyên epoch

    # Trục Y bên phải: Perplexity (PPL)
    ax2 = ax1.twinx()
    sns.lineplot(data=df_val, x="epoch", y="perplexity", 
                 ax=ax2, color=COLORS["ppl"], marker="s", linestyle="--", linewidth=2, label="Perplexity (PPL)")
    ax2.set_ylabel("Perplexity (Lower is Better)", color=COLORS["ppl"], fontweight="bold")
    ax2.tick_params(axis='y', labelcolor=COLORS["ppl"])
    
    plt.title("Validation Metrics per Epoch", fontweight="bold")
    
    # Tạo chú thích (Legend) chung cho cả 2 trục
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "val_metrics.png"), dpi=300)
    print(f"Saved: {OUTPUT_DIR}/val_metrics.png")

if __name__ == "__main__":
    # 1. Đọc dữ liệu từ CSV
    try:
        train_df = pd.read_csv(os.path.join(LOG_DIR, "train_logs.csv"))
        val_df = pd.read_csv(os.path.join(LOG_DIR, "val_logs.csv"))
        print("Logs loaded successfully.")
    except FileNotFoundError:
        print("Error: CSV logs not found in 'checkpoints/'. Did you run training?")
        exit()

    # 2. Vẽ các hình
    print("Generating figures...")
    plot_training_loss(train_df)
    plot_learning_rate(train_df)
    plot_validation_metrics(val_df)
    print("Done!")