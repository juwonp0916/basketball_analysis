import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('backend/results/video1_full_report.json', 'r') as f:
    data = json.load(f)

# Create 4 visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Detection Rates
ax1 = axes[0, 0]
categories = ['Ball', 'Rim', 'Both', 'Person', 'Shoot']
detections = [data['ball_detected'], data['rim_detected'], data['both_detected'],
              data['person_detected'], data['shoot_detected']]
total = [1227, 1227, 1227, 245, 245]  # Based on frames_processed and sampling
rates = [d/t*100 for d, t in zip(detections, total)]
ax1.bar(categories, rates, color=['orange', 'blue', 'green', 'purple', 'red'])
ax1.set_ylabel('Detection Rate (%)')
ax1.set_title('Detection Rates - video1.mp4')
ax1.set_ylim(0, 100)
for i, (cat, rate) in enumerate(zip(categories, rates)):
    ax1.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=10)

# 2. Ball Confidence Distribution
ax2 = axes[0, 1]
ax2.hist(data['ball_confidences'], bins=30, color='orange', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Confidence Score')
ax2.set_ylabel('Frequency')
avg_ball = np.mean(data['ball_confidences'])
ax2.set_title(f'Ball Detection Confidence (avg: {avg_ball:.3f})')
ax2.axvline(avg_ball, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_ball:.3f}')
ax2.legend()

# 3. Rim Confidence Distribution
ax3 = axes[1, 0]
ax3.hist(data['rim_confidences'], bins=30, color='blue', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Confidence Score')
ax3.set_ylabel('Frequency')
avg_rim = np.mean(data['rim_confidences'])
ax3.set_title(f'Rim Detection Confidence (avg: {avg_rim:.3f})')
ax3.axvline(avg_rim, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_rim:.3f}')
ax3.legend()

# 4. Confidence Over Time (Ball)
ax4 = axes[1, 1]
ax4.plot(data['ball_confidences'], color='orange', alpha=0.6, linewidth=0.8)
ax4.set_xlabel('Frame Number')
ax4.set_ylabel('Confidence')
ax4.set_title('Ball Detection Confidence Over Time')
ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
ax4.axhline(avg_ball, color='green', linestyle='--', alpha=0.5, label=f'Mean ({avg_ball:.3f})')
ax4.legend()
ax4.set_ylim(0, 1)

plt.tight_layout()

# Save the figure
output_file = 'backend/results/video1_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Visualization saved to: {output_file}")

# Print summary statistics
print("\n" + "="*60)
print("DETECTION SUMMARY - video1.mp4")
print("="*60)
print(f"Frames Analyzed: {data['frames_processed']} / {data['video_info']['total_frames']} total")
print(f"\nDetection Rates:")
print(f"  Ball:   {data['ball_detected']:4d} / {data['frames_processed']} ({data['ball_detected']/data['frames_processed']*100:5.1f}%)")
print(f"  Rim:    {data['rim_detected']:4d} / {data['frames_processed']} ({data['rim_detected']/data['frames_processed']*100:5.1f}%)")
print(f"  Both:   {data['both_detected']:4d} / {data['frames_processed']} ({data['both_detected']/data['frames_processed']*100:5.1f}%)")
print(f"  Person: {data['person_detected']:4d} / 245 sampled ({data['person_detected']/245*100:5.1f}%)")
print(f"  Shoot:  {data['shoot_detected']:4d} / 245 sampled ({data['shoot_detected']/245*100:5.1f}%)")

print(f"\nAverage Confidence Scores:")
print(f"  Ball:   {avg_ball:.3f}")
print(f"  Rim:    {avg_rim:.3f}")
print(f"  Person: {np.mean(data['person_confidences']):.3f}")

print(f"\nConfidence Ranges:")
print(f"  Ball:   {min(data['ball_confidences']):.3f} - {max(data['ball_confidences']):.3f}")
print(f"  Rim:    {min(data['rim_confidences']):.3f} - {max(data['rim_confidences']):.3f}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
ball_rate = data['ball_detected']/data['frames_processed']*100
if ball_rate < 50:
    print("[!] Ball detection (36.1%) is BELOW expected range (55-70%)")
    print("  -> Fine-tuning is ESSENTIAL")
elif ball_rate < 70:
    print("[!] Ball detection is moderate - fine-tuning recommended")
else:
    print("[OK] Ball detection is good")

rim_rate = data['rim_detected']/data['frames_processed']*100
if rim_rate > 90:
    print("[OK] Rim detection (99.9%) is EXCELLENT")
else:
    print("[!] Rim detection needs improvement")

print("="*60)
