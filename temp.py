import pickle
import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
marker_list = ['s', 'o', '^', 'v']
color_list = ['k','r','g','b']


with open('parameter_only_results_rho_vs_u.pkl', 'rb') as f:
    gcn_results_rho_vs_u = pickle.load(f)
with open('parameter_only_results_rho_vs_w.pkl', 'rb') as f:
    gcn_results_rho_vs_w = pickle.load(f)

print(f"Total GCN results for rho_vs_u: {len(gcn_results_rho_vs_u)}")
print(f"Total GCN results for rho_vs_w: {len(gcn_results_rho_vs_w)}")

        # Plot GCN results for left panel
gcn_w_values = sorted(set([round(result[1], 5) for result in gcn_results_rho_vs_u]))
i = 0
for w in gcn_w_values:
    u_values = [round(result[0], 5) for result in gcn_results_rho_vs_u if round(result[1], 5) == w]
    rho_values = [round(result[2], 5) for result in gcn_results_rho_vs_u if round(result[1], 5) == w]
    if u_values:
        ax1.plot(u_values, rho_values, linestyle='--',color = color_list[i], label=f'w = {w} (NN)')
        print(f"GCN data for w={w}: {len(u_values)} points")
        i += 1
    else:
        print(f"No GCN data available for w={w}")
    

        # Plot GCN results for right panel
gcn_u_values = sorted(set([round(result[0], 5) for result in gcn_results_rho_vs_w]))
i = 0
for u in gcn_u_values:
    w_values = [round(result[1], 5) for result in gcn_results_rho_vs_w if round(result[0], 5) == u]
    rho_values = [round(result[2], 5) for result in gcn_results_rho_vs_w if round(result[0], 5) == u]
    if w_values:
        ax2.plot(w_values, rho_values, linestyle='--',color = color_list[i], label=f'u = {u} (NN)')
        print(f"GCN data for u={u}: {len(w_values)} points")
        i += 1
    else:
        print(f"No GCN data available for u={u}")
    

# Set labels and titles
ax1.set_xlabel('Cost-to-benefit ratio, u')
ax1.set_ylabel('Fraction of cooperators')
ax1.set_title('Cooperators vs Cost-to-benefit ratio')
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.01, 1.01)
ax1.legend()

ax2.set_xlabel('Strategy updating probability, w')
ax2.set_ylabel('Fraction of cooperators')
ax2.set_title('Cooperators vs Strategy updating probability')
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.01, 1.01)
ax2.legend()
plt.tight_layout()
plt.show()
print("Finish")
