import matplotlib.pyplot as plt

percent_x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

val_simsiam_y = [0.3463, 0.3412, 0.3490, 0.3415, 0.3328, 0.3412, 0.3410, 0.3421, 0.3333, 0.3427, 0.3431, 0.3451, 0.3416, 0.3446, 0.3352, 0.3386, 0.3402, 0.3478, 0.0388, 0.0388]
robust_simsiam_y = [0.2435, 0.2459, 0.2420, 0.2455, 0.2392, 0.2593, 0.2497, 0.2430, 0.2366, 0.2303, 0.2342, 0.2456, 0.2501, 0.2589, 0.2399, 0.2366, 0.2597, 0.2440, 0.0367, 0.0367]


val_rankneg_y = [0.3449, 0.3600, 0.3452, 0.3418, 0.3336, 0.3401, 0.3339, 0.3410, 0.3384, 0.3416, 0.3433, 0.3397, 0.3370, 0.3495, 0.3328, 0.3500, 0.3496, 0.3442, 0.2818, 0.2893]
robust_rankneg_y = [0.2460, 0.2590, 0.2504, 0.2501, 0.2478, 0.2342, 0.2415, 0.2475, 0.2328, 0.2407, 0.2437, 0.2330, 0.2360, 0.2549, 0.2382, 0.2450, 0.2444, 0.2405, 0.1784, 0.2105]


# plt.plot(percent_x, val_simsiam_y, label='SimSiam')
# plt.plot(percent_x, val_rankneg_y, label='RankNeg')
# plt.legend()
# plt.xlabel('Corruption Ratio')
# plt.ylabel('Validation NDCG')
# plt.savefig('/afs/ece.cmu.edu/usr/charlieh/ptranking/job_submit/val_ablation_plot.png')

plt.plot(percent_x, robust_simsiam_y, label='SimSiam')
plt.plot(percent_x, robust_rankneg_y, label='RankNeg')
plt.legend()
plt.xlabel('Corruption Ratio')
plt.ylabel('Validation Robust NDCG')
plt.savefig('/afs/ece.cmu.edu/usr/charlieh/ptranking/job_submit/robust_ablation_plot.png')
