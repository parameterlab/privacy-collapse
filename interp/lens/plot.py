import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data from your CSV file
# Replace 'your_data.csv' with the actual path to your file
base = pd.read_csv('base_trajectory_results.csv')
ft = pd.read_csv('finetuned_trajectory_results.csv')

# 2. Create the plot
plt.figure(figsize=(10, 6))

# Plotting the mean ranks (you can also add medians if desired)
plt.plot(base['layer'], (base['prob_correct_mean'] - base['prob_wrong_mean']), marker='o', linestyle='-', label='Base Model', color='blue')
plt.plot(ft['layer'], (ft['prob_correct_mean'] - ft['prob_wrong_mean']), marker='s', linestyle='--', label='Fine-tuned Model', color='orange')

# plt.plot(base['layer'], base['rank_wrong_median'], marker='o', linestyle='-', label='Base Model', color='blue')
# plt.plot(ft['layer'], ft['rank_wrong_median'], marker='s', linestyle='--', label='Fine-tuned Model', color='orange')

# 3. Apply Log Scale to the Y-axis
# plt.yscale('log')

# 4. Add labels and styling
plt.xlabel('Layer', fontsize=12)
# plt.ylabel('Rank (Log Scale)', fontsize=12)
# plt.title('Log Rank vs. Layer', fontsize=14)
plt.legend()
# plt.grid(True, which="both", ls="-", alpha=0.3)

# 5. Show or save the plot
plt.savefig('log_rank_plot.png')
plt.show()