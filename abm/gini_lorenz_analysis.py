# gini_lorenz_analysis.py

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mesa import Agent, Model
from mesa.time import BaseScheduler
from matplotlib.lines import Line2D

# --- USER CONFIGURATION ---
OUTPUT_DIR = "outputs3"
NUM_WORKERS = 100
NUM_GIGS = 50
STEPS = 100

PERCENT_DISCRIMINATORY_GIGS = 0.4
PERCENT_HIGH_PERFORMANCE = 0.4
PERCENT_HIGH_SCHEDULING = 0.0
PERCENT_SPOILED_IDENTITY = 0.5
SCHEDULING_THRESHOLD = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODEL CODE ---

class GigAssignment:
    def __init__(self, gig_id, pay, discriminatory_attitude):
        self.gig_id = gig_id
        self.pay = pay
        self.discriminatory_attitude = discriminatory_attitude
        self.taken = False

    def rate_worker(self, worker):
        base_rating = random.uniform(3.0, 5.0)
        if self.discriminatory_attitude > 0.5 and worker.spoiled_identity:
            base_rating -= self.discriminatory_attitude * random.uniform(0.5, 1.5)
        if worker.performance_ability > 0.7:
            base_rating += random.uniform(0.5, 1.5)
        base_rating = max(1.0, min(5.0, base_rating))
        return base_rating

class WorkerAgent(Agent):
    def __init__(self, unique_id, model, spoiled_identity=False, performance_ability=0.5, scheduling=0.5):
        super().__init__(unique_id, model)
        self.spoiled_identity = spoiled_identity
        self.performance_ability = performance_ability
        self.scheduling = scheduling
        self.rating_history = []
        self.pay_history = []
        self.spell_streak = 0
        self.precarity_history = []
        self.choices_history = []

    @property
    def rating(self):
        avg_rating = np.mean(self.rating_history) if self.rating_history else 5.0
        penalty = 1.0 if self.spell_streak >= 3 else 0.0
        return max(1.0, avg_rating - penalty)

    def step(self, available_gigs):
        self.choices_history.append(len(available_gigs))
        skip_prob = 0.7 if self.scheduling > SCHEDULING_THRESHOLD else 0.1
        if random.random() < skip_prob:
            self.spell_streak += 1
            self.pay_history.append(0)
            prev_rating = self.rating_history[-1] if self.rating_history else 5.0
            penalty = 1.0 if self.spell_streak >= 3 else 0.0
            self.rating_history.append(max(1.0, prev_rating - penalty))
            return

        if available_gigs:
            chosen_gig = max(available_gigs, key=lambda g: g.pay)
            available_gigs.remove(chosen_gig)
            chosen_gig.taken = True
            rating = chosen_gig.rate_worker(self)
            self.rating_history.append(rating)
            self.pay_history.append(chosen_gig.pay)
            self.spell_streak = 0
        else:
            self.spell_streak += 1
            self.pay_history.append(0)
            prev_rating = self.rating_history[-1] if self.rating_history else 5.0
            penalty = 1.0 if self.spell_streak >= 3 else 0.0
            self.rating_history.append(max(1.0, prev_rating - penalty))

class PeopleWorkModel(Model):
    def __init__(self, num_workers=NUM_WORKERS, num_gigs=NUM_GIGS):
        self.schedule = BaseScheduler(self)
        self.workers = []
        self.gigs = []

        for i in range(num_workers):
            spoiled = random.random() < PERCENT_SPOILED_IDENTITY
            perf = random.uniform(0.8, 1.0) if random.random() < PERCENT_HIGH_PERFORMANCE else random.uniform(0, 0.7)
            scheduling = random.uniform(0.8, 1.0) if random.random() < PERCENT_HIGH_SCHEDULING else random.uniform(0, 0.7)
            worker = WorkerAgent(i, self, spoiled_identity=spoiled, performance_ability=perf, scheduling=scheduling)
            self.workers.append(worker)
            self.schedule.add(worker)

        num_disc_gigs = int(PERCENT_DISCRIMINATORY_GIGS * num_gigs)
        for i in range(num_gigs):
            pay = random.uniform(0, 1)
            if i < num_disc_gigs:
                disc_att = random.uniform(0.5, 1.0)
            else:
                disc_att = random.uniform(0.0, 0.5)
            gig = GigAssignment(gig_id=i, pay=pay, discriminatory_attitude=disc_att)
            self.gigs.append(gig)

    def step(self):
        for gig in self.gigs:
            gig.taken = False

        sorted_workers = sorted(self.workers, key=lambda w: w.rating, reverse=True)
        available_gigs = [gig for gig in self.gigs if not gig.taken]
        for worker in sorted_workers:
            choices = [g for g in available_gigs]
            worker.step(choices)
            available_gigs = [g for g in available_gigs if not g.taken]

# --- Variant B: No penalty for missing gigs ---
class WorkerAgentNoPenalty(WorkerAgent):
    @property
    def rating(self):
        avg_rating = np.mean(self.rating_history) if self.rating_history else 5.0
        # No penalty for spell streak
        return avg_rating

# --- Variant C: Order by performance ability, not rating ---
class PeopleWorkModelRandomOrder(PeopleWorkModel):
    def step(self):
        for gig in self.gigs:
            gig.taken = False

        # Random order each step
        shuffled_workers = self.workers[:]
        random.shuffle(shuffled_workers)
        available_gigs = [gig for gig in self.gigs if not gig.taken]
        for worker in shuffled_workers:
            choices = [g for g in available_gigs]
            worker.step(choices)
            available_gigs = [g for g in available_gigs if not g.taken]

# --- Variant B Model (uses WorkerAgentNoPenalty) ---
class PeopleWorkModelNoPenalty(PeopleWorkModel):
    def __init__(self, num_workers=NUM_WORKERS, num_gigs=NUM_GIGS):
        self.schedule = BaseScheduler(self)
        self.workers = []
        self.gigs = []

        for i in range(num_workers):
            spoiled = random.random() < PERCENT_SPOILED_IDENTITY
            perf = random.uniform(0.8, 1.0) if random.random() < PERCENT_HIGH_PERFORMANCE else random.uniform(0, 0.7)
            scheduling = random.uniform(0.8, 1.0) if random.random() < PERCENT_HIGH_SCHEDULING else random.uniform(0, 0.7)
            worker = WorkerAgentNoPenalty(i, self, spoiled_identity=spoiled, performance_ability=perf, scheduling=scheduling)
            self.workers.append(worker)
            self.schedule.add(worker)

        num_disc_gigs = int(PERCENT_DISCRIMINATORY_GIGS * num_gigs)
        for i in range(num_gigs):
            pay = random.uniform(0, 1)
            if i < num_disc_gigs:
                disc_att = random.uniform(0.5, 1.0)
            else:
                disc_att = random.uniform(0.0, 0.5)
            gig = GigAssignment(gig_id=i, pay=pay, discriminatory_attitude=disc_att)
            self.gigs.append(gig)

# --- Gini and Lorenz functions ---

def gini(array):
    array = np.array(array, dtype=float)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-10
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def lorenz_curve(array):
    array = np.array(array, dtype=float)
    array = np.sort(array)
    cum_array = np.cumsum(array)
    total = cum_array[-1]
    x_lorenz = np.linspace(0, 1, len(array) + 1)
    y_lorenz = np.concatenate([[0], cum_array / total])
    return x_lorenz, y_lorenz

def theil_index(x):
    x = np.array(x, dtype=float)
    x = x[x > 0]  # avoid log(0)
    mean = np.mean(x)
    return np.mean(x * np.log(x / mean)) / mean

def theil_decomposition(values, group_labels):
    """Returns (total, within, between) Theil index for values, grouped by group_labels."""
    values = np.array(values, dtype=float)
    group_labels = np.array(group_labels)
    overall_mean = np.mean(values)
    unique_groups = np.unique(group_labels)
    n = len(values)
    theil_within = 0
    theil_between = 0
    for group in unique_groups:
        group_vals = values[group_labels == group]
        group_n = len(group_vals)
        if group_n == 0:
            continue
        group_mean = np.mean(group_vals)
        theil_within += (group_n / n) * theil_index(group_vals)
        theil_between += (group_n / n) * (group_mean / overall_mean) * np.log(group_mean / overall_mean) if group_mean > 0 else 0
    theil_total = theil_within + theil_between
    return theil_total, theil_within, theil_between

# --- Run Model ---
if __name__ == "__main__":
    model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
    for i in range(STEPS):
        model.step()

    # --- Cumulative Pay ---
    cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
    gini_pay = gini(cumulative_pays)
    print(f"Gini coefficient for cumulative pay: {gini_pay:.3f}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(cumulative_pays)), sorted(cumulative_pays), color='purple', alpha=0.7)
    plt.xlabel("Worker (sorted by cumulative pay)")
    plt.ylabel("Cumulative Pay")
    plt.title(f"Cumulative Pay Distribution (Gini: {gini_pay:.3f})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_pay_distribution_gini.png")
    plt.close()

    x_lorenz, y_lorenz = lorenz_curve(cumulative_pays)
    plt.figure(figsize=(8, 6))
    plt.plot(x_lorenz, y_lorenz, label="Lorenz Curve", color="purple")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Line of Equality")
    plt.xlabel("Cumulative Share of Workers")
    plt.ylabel("Cumulative Share of Pay")
    plt.title(f"Lorenz Curve of Cumulative Pay (Gini: {gini_pay:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lorenz_curve_cumulative_pay.png")
    plt.close()

    # --- Jobs Presented ---
    jobs_presented = [np.sum(worker.choices_history) for worker in model.workers]
    gini_jobs = gini(jobs_presented)
    print(f"Gini coefficient for jobs presented: {gini_jobs:.3f}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(jobs_presented)), sorted(jobs_presented), color='teal', alpha=0.7)
    plt.xlabel("Worker (sorted by jobs presented)")
    plt.ylabel("Total Jobs Presented")
    plt.title(f"Jobs Presented Distribution (Gini: {gini_jobs:.3f})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/jobs_presented_distribution_gini.png")
    plt.close()

    x_lorenz, y_lorenz = lorenz_curve(jobs_presented)
    plt.figure(figsize=(8, 6))
    plt.plot(x_lorenz, y_lorenz, label="Lorenz Curve", color="teal")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Line of Equality")
    plt.xlabel("Cumulative Share of Workers")
    plt.ylabel("Cumulative Share of Jobs Presented")
    plt.title(f"Lorenz Curve of Jobs Presented (Gini: {gini_jobs:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lorenz_curve_jobs_presented.png")
    plt.close()



model_variants = [
    ("Rating w/ Penalty", PeopleWorkModel),
    ("Rating w/o Penalty", PeopleWorkModelNoPenalty),
    ("Random Order w/ Penalty", PeopleWorkModelRandomOrder),
]

gini_results = []
lorenz_results = []

for label, ModelClass in model_variants:
    model = ModelClass(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
    for i in range(STEPS):
        model.step()
    cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
    gini_val = gini(cumulative_pays)
    gini_results.append((label, gini_val))
    x_lorenz, y_lorenz = lorenz_curve(cumulative_pays)
    lorenz_results.append((label, x_lorenz, y_lorenz))

# --- Plot Gini coefficients ---
plt.figure(figsize=(7, 5))
labels, ginis = zip(*gini_results)
plt.bar(labels, ginis, color=['purple', 'orange', 'green'])
plt.ylabel("Gini Coefficient (Cumulative Pay)")
plt.title("Gini Coefficient Comparison Across Models")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gini_comparison_models.png")
plt.close()

# --- Plot Lorenz curves ---
plt.figure(figsize=(7, 7))
for label, x_lorenz, y_lorenz in lorenz_results:
    plt.plot(x_lorenz, y_lorenz, label=label)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Line of Equality")
plt.xlabel("Cumulative Share of Workers")
plt.ylabel("Cumulative Share of Pay")
plt.title("Lorenz Curves: Cumulative Pay Across Models")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lorenz_curves_comparison_models.png")
plt.close()

import seaborn as sns

for label, ModelClass in model_variants:
    model = ModelClass(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
    for i in range(STEPS):
        model.step()

    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        running_avg_pay = np.cumsum(worker.pay_history) / (np.arange(len(worker.pay_history)) + 1)
        plt.plot(running_avg_pay, color=color, linestyle=linestyle, alpha=0.7)
    # Legend elements
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Spoiled Identity'),
        Line2D([0], [0], color='blue', lw=2, label='Non-Spoiled Identity'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='High Performance (>0.7)'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Low Performance (≤0.7)'),
    ]
    plt.xlabel("Step")
    plt.ylabel("Cumulative Average Pay")
    plt.title(f"Cumulative Average Pay Over Time\n{label}")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    fname = label.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(f"{OUTPUT_DIR}/cumulative_avg_pay_over_time_{fname}.png")
    plt.close()

import seaborn as sns

group_names = {
    "spoiled": "Spoiled",
    "nonspoiled": "Non-Spoiled",
    "high_perf": "High Perf",
    "low_perf": "Low Perf",
    "high_sched": "High Sched",
    "low_sched": "Low Sched"
}

for label, ModelClass in model_variants:
    model = ModelClass(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
    for i in range(STEPS):
        model.step()
    cumulative_pays = np.array([np.sum(w.pay_history) for w in model.workers])
    spoiled = np.array([w.spoiled_identity for w in model.workers])
    high_perf = np.array([w.performance_ability > 0.7 for w in model.workers])
    high_sched = np.array([w.scheduling > SCHEDULING_THRESHOLD for w in model.workers])

    # --- Theil index decomposition ---
    group_labels_spoiled = np.where(spoiled, "spoiled", "nonspoiled")
    group_labels_perf = np.where(high_perf, "high_perf", "low_perf")
    group_labels_sched = np.where(high_sched, "high_sched", "low_sched")

    theil_spoiled = theil_decomposition(cumulative_pays, group_labels_spoiled)
    theil_perf = theil_decomposition(cumulative_pays, group_labels_perf)
    theil_sched = theil_decomposition(cumulative_pays, group_labels_sched)

    print(f"\n{label}:")
    print(f"  Theil (spoiled): total={theil_spoiled[0]:.3f}, within={theil_spoiled[1]:.3f}, between={theil_spoiled[2]:.3f}")
    print(f"  Theil (performance): total={theil_perf[0]:.3f}, within={theil_perf[1]:.3f}, between={theil_perf[2]:.3f}")
    print(f"  Theil (scheduling): total={theil_sched[0]:.3f}, within={theil_sched[1]:.3f}, between={theil_sched[2]:.3f}")

    # --- Lorenz curves by group ---
    plt.figure(figsize=(8, 6))
    for group, mask, color in [
        ("Spoiled", spoiled, "red"),
        ("Non-Spoiled", ~spoiled, "blue"),
        ("High Perf", high_perf, "black"),
        ("Low Perf", ~high_perf, "gray"),
        ("High Sched", high_sched, "green"),
        ("Low Sched", ~high_sched, "orange"),
    ]:
        vals = cumulative_pays[mask]
        if len(vals) > 1:
            x_lorenz, y_lorenz = lorenz_curve(vals)
            plt.plot(x_lorenz, y_lorenz, label=group, color=color)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Line of Equality")
    plt.xlabel("Cumulative Share of Workers")
    plt.ylabel("Cumulative Share of Pay")
    plt.title(f"Lorenz Curves by Group\n{label}")
    plt.legend()
    plt.tight_layout()
    fname = label.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(f"{OUTPUT_DIR}/lorenz_curves_by_group_{fname}.png")
    plt.close()

    # --- Boxplots by group ---
    # Spoiled vs Non-Spoiled
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=group_labels_spoiled, y=cumulative_pays, palette={"spoiled": "red", "nonspoiled": "blue"})
    plt.title(f"Boxplot: Spoiled vs Non-Spoiled\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Pay")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplot_spoiled_{fname}.png")
    plt.close()

    # High vs Low Performance
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=group_labels_perf, y=cumulative_pays, palette={"high_perf": "black", "low_perf": "gray"})
    plt.title(f"Boxplot: High vs Low Performance\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Pay")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplot_performance_{fname}.png")
    plt.close()

    # High vs Low Scheduling
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=group_labels_sched, y=cumulative_pays, palette={"high_sched": "green", "low_sched": "orange"})
    plt.title(f"Boxplot: High vs Low Scheduling\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Pay")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplot_scheduling_{fname}.png")
    plt.close()

    # --- Violin plots by group ---
    # Spoiled vs Non-Spoiled
    plt.figure(figsize=(7, 5))
    sns.violinplot(x=group_labels_spoiled, y=cumulative_pays, palette={"spoiled": "red", "nonspoiled": "blue"})
    plt.title(f"Violin Plot: Spoiled vs Non-Spoiled\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Pay")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/violin_spoiled_{fname}.png")
    plt.close()

    # High vs Low Performance
    plt.figure(figsize=(7, 5))
    sns.violinplot(x=group_labels_perf, y=cumulative_pays, palette={"high_perf": "black", "low_perf": "gray"})
    plt.title(f"Violin Plot: High vs Low Performance\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Pay")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/violin_performance_{fname}.png")
    plt.close()

    # High vs Low Scheduling
    plt.figure(figsize=(7, 5))
    sns.violinplot(x=group_labels_sched, y=cumulative_pays, palette={"high_sched": "green", "low_sched": "orange"})
    plt.title(f"Violin Plot: High vs Low Scheduling\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Pay")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/violin_scheduling_{fname}.png")
    plt.close()

for label, ModelClass in model_variants:
    model = ModelClass(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
    for i in range(STEPS):
        model.step()
    jobs_presented = np.array([np.sum(w.choices_history) for w in model.workers])
    spoiled = np.array([w.spoiled_identity for w in model.workers])
    high_perf = np.array([w.performance_ability > 0.7 for w in model.workers])
    high_sched = np.array([w.scheduling > SCHEDULING_THRESHOLD for w in model.workers])

    group_labels_spoiled = np.where(spoiled, "spoiled", "nonspoiled")
    group_labels_perf = np.where(high_perf, "high_perf", "low_perf")
    group_labels_sched = np.where(high_sched, "high_sched", "low_sched")

    # Spoiled vs Non-Spoiled
    plt.figure(figsize=(7, 5))
    sns.violinplot(x=group_labels_spoiled, y=jobs_presented, palette={"spoiled": "red", "nonspoiled": "blue"})
    plt.title(f"Violin Plot: Jobs Presented - Spoiled vs Non-Spoiled\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Jobs Presented")
    plt.tight_layout()
    fname = label.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(f"{OUTPUT_DIR}/violin_jobs_spoiled_{fname}.png")
    plt.close()

    # High vs Low Performance
    plt.figure(figsize=(7, 5))
    sns.violinplot(x=group_labels_perf, y=jobs_presented, palette={"high_perf": "black", "low_perf": "gray"})
    plt.title(f"Violin Plot: Jobs Presented - High vs Low Performance\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Jobs Presented")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/violin_jobs_performance_{fname}.png")
    plt.close()

    # High vs Low Scheduling
    plt.figure(figsize=(7, 5))
    sns.violinplot(x=group_labels_sched, y=jobs_presented, palette={"high_sched": "green", "low_sched": "orange"})
    plt.title(f"Violin Plot: Jobs Presented - High vs Low Scheduling\n{label}")
    plt.xlabel("Group")
    plt.ylabel("Cumulative Jobs Presented")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/violin_jobs_scheduling_{fname}.png")
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

spoiled_grid = np.linspace(0, 1, 11)      # 0.0, 0.1, ..., 1.0
performance_grid = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
gini_matrix = np.zeros((len(spoiled_grid), len(performance_grid)))

for i, spoiled_frac in enumerate(spoiled_grid):
    for j, perf_frac in enumerate(performance_grid):
        # Set parameters for this run
        PERCENT_SPOILED_IDENTITY = spoiled_frac
        PERCENT_HIGH_PERFORMANCE = perf_frac

        # Run the model (reduce NUM_WORKERS and STEPS for speed if needed)
        model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
        for step in range(STEPS):
            model.step()
        cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
        gini_matrix[i, j] = gini(cumulative_pays)

plt.figure(figsize=(8, 6))
im = plt.imshow(gini_matrix, origin='lower', aspect='auto', cmap='viridis',
                extent=[performance_grid[0], performance_grid[-1], spoiled_grid[0], spoiled_grid[-1]])
plt.colorbar(im, label="Gini Coefficient (Cumulative Pay)")
plt.xlabel("Fraction of Workers with High Performance")
plt.ylabel("Fraction of Workers with Spoiled Identity")
plt.title("Gini Coefficient Heatmap\n(Cumulative Pay vs. Spoiled Identity & High Performance)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_gini_spoiled_vs_performance.png")
plt.close()

disc_grid = np.linspace(0, 1, 11)      # 0.0, 0.1, ..., 1.0
sched_grid = np.linspace(0, 1, 11)     # 0.0, 0.1, ..., 1.0
gini_matrix_sched = np.zeros((len(disc_grid), len(sched_grid)))

for i, disc_frac in enumerate(disc_grid):
    for j, sched_frac in enumerate(sched_grid):
        # Set parameters for this run
        PERCENT_DISCRIMINATORY_GIGS = disc_frac
        PERCENT_HIGH_SCHEDULING = sched_frac

        # Run the model (reduce NUM_WORKERS and STEPS for speed if needed)
        model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
        for step in range(STEPS):
            model.step()
        cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
        gini_matrix_sched[i, j] = gini(cumulative_pays)

plt.figure(figsize=(8, 6))
im = plt.imshow(gini_matrix_sched, origin='lower', aspect='auto', cmap='viridis',
                extent=[sched_grid[0], sched_grid[-1], disc_grid[0], disc_grid[-1]])
plt.colorbar(im, label="Gini Coefficient (Cumulative Pay)")
plt.xlabel("Fraction of Workers with High Scheduling")
plt.ylabel("Fraction of Discriminatory Gigs")
plt.title("Gini Coefficient Heatmap\n(Cumulative Pay vs. Discriminatory Gigs & High Scheduling)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_gini_discriminatory_vs_scheduling.png")
plt.close()


import numpy as np
import matplotlib.pyplot as plt

# Define grids
spoiled_grid = np.linspace(0, 1, 6)      # e.g. 0.0, 0.2, ..., 1.0
performance_grid = np.linspace(0, 1, 6)
scheduling_grid = np.linspace(0, 1, 6)

gini_cube = np.zeros((len(spoiled_grid), len(performance_grid), len(scheduling_grid)))

for i, spoiled_frac in enumerate(spoiled_grid):
    for j, perf_frac in enumerate(performance_grid):
        for k, sched_frac in enumerate(scheduling_grid):
            # Set parameters for this run
            PERCENT_SPOILED_IDENTITY = spoiled_frac
            PERCENT_HIGH_PERFORMANCE = perf_frac
            PERCENT_HIGH_SCHEDULING = sched_frac

            model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
            for step in range(STEPS):
                model.step()
            cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
            gini_cube[i, j, k] = gini(cumulative_pays)

# Visualize as a grid of 2D heatmaps for each scheduling value
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
for idx, sched_frac in enumerate(scheduling_grid):
    ax = axes.flat[idx]
    im = ax.imshow(
        gini_cube[:, :, idx],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        extent=[performance_grid[0], performance_grid[-1], spoiled_grid[0], spoiled_grid[-1]]
    )
    ax.set_title(f"High Scheduling = {sched_frac:.2f}")
    ax.set_xlabel("Fraction High Performance")
    ax.set_ylabel("Fraction Spoiled Identity")
fig.colorbar(im, ax=axes.ravel().tolist(), label="Gini Coefficient (Cumulative Pay)")
plt.suptitle("Gini Coefficient Slices by High Scheduling")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUTPUT_DIR}/gini_3d_heatmap_slices.png")
plt.close()


sched_grid = np.linspace(0, 1, 11)      # 0.0, 0.1, ..., 1.0
spoiled_grid = np.linspace(0, 1, 11)
gini_matrix_sched_spoiled = np.zeros((len(spoiled_grid), len(sched_grid)))

for i, spoiled_frac in enumerate(spoiled_grid):
    for j, sched_frac in enumerate(sched_grid):
        PERCENT_SPOILED_IDENTITY = spoiled_frac
        PERCENT_HIGH_SCHEDULING = sched_frac

        model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
        for step in range(STEPS):
            model.step()
        cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
        gini_matrix_sched_spoiled[i, j] = gini(cumulative_pays)

plt.figure(figsize=(8, 6))
im = plt.imshow(gini_matrix_sched_spoiled, origin='lower', aspect='auto', cmap='viridis',
                extent=[sched_grid[0], sched_grid[-1], spoiled_grid[0], spoiled_grid[-1]])
plt.colorbar(im, label="Gini Coefficient (Cumulative Pay)")
plt.xlabel("Fraction of Workers with High Scheduling")
plt.ylabel("Fraction of Workers with Spoiled Identity")
plt.title("Gini Coefficient Heatmap\n(Cumulative Pay vs. High Scheduling & Spoiled Identity)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_gini_scheduling_vs_spoiled.png")
plt.close()

disc_grid = np.linspace(0, 1, 11)      # 0.0, 0.1, ..., 1.0
spoiled_grid = np.linspace(0, 1, 11)
gini_matrix_disc_spoiled = np.zeros((len(spoiled_grid), len(disc_grid)))

for i, spoiled_frac in enumerate(spoiled_grid):
    for j, disc_frac in enumerate(disc_grid):
        PERCENT_SPOILED_IDENTITY = spoiled_frac
        PERCENT_DISCRIMINATORY_GIGS = disc_frac

        model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
        for step in range(STEPS):
            model.step()
        cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
        gini_matrix_disc_spoiled[i, j] = gini(cumulative_pays)

plt.figure(figsize=(8, 6))
im = plt.imshow(gini_matrix_disc_spoiled, origin='lower', aspect='auto', cmap='viridis',
                extent=[disc_grid[0], disc_grid[-1], spoiled_grid[0], spoiled_grid[-1]])
plt.colorbar(im, label="Gini Coefficient (Cumulative Pay)")
plt.xlabel("Fraction of Discriminatory Gigs")
plt.ylabel("Fraction of Workers with Spoiled Identity")
plt.title("Gini Coefficient Heatmap\n(Cumulative Pay vs. Discriminatory Gigs & Spoiled Identity)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_gini_discriminatory_vs_spoiled.png")
plt.close()


gig_worker_ratios = np.linspace(0.1, 1.0, 10)  # 10% to 100%
spoiled_grid = np.linspace(0, 1, 11)
gini_matrix_gigworker_spoiled = np.zeros((len(spoiled_grid), len(gig_worker_ratios)))

for i, spoiled_frac in enumerate(spoiled_grid):
    for j, ratio in enumerate(gig_worker_ratios):
        PERCENT_SPOILED_IDENTITY = spoiled_frac
        num_gigs = int(NUM_WORKERS * ratio)
        model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=num_gigs)
        for step in range(STEPS):
            model.step()
        cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
        gini_matrix_gigworker_spoiled[i, j] = gini(cumulative_pays)

plt.figure(figsize=(8, 6))
im = plt.imshow(gini_matrix_gigworker_spoiled, origin='lower', aspect='auto', cmap='viridis',
                extent=[gig_worker_ratios[0], gig_worker_ratios[-1], spoiled_grid[0], spoiled_grid[-1]])
plt.colorbar(im, label="Gini Coefficient (Cumulative Pay)")
plt.xlabel("Ratio of Gigs to Workers")
plt.ylabel("Fraction of Workers with Spoiled Identity")
plt.title("Gini Coefficient Heatmap\n(Cumulative Pay vs. Gigs/Workers Ratio & Spoiled Identity)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_gini_gigworker_vs_spoiled.png")
plt.close()

gig_worker_ratios = np.linspace(0.1, 1.0, 10)  # 10% to 100%
sched_grid = np.linspace(0, 1, 11)
gini_matrix_gigworker_sched = np.zeros((len(sched_grid), len(gig_worker_ratios)))

for i, sched_frac in enumerate(sched_grid):
    for j, ratio in enumerate(gig_worker_ratios):
        PERCENT_HIGH_SCHEDULING = sched_frac
        num_gigs = int(NUM_WORKERS * ratio)
        model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=num_gigs)
        for step in range(STEPS):
            model.step()
        cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]
        gini_matrix_gigworker_sched[i, j] = gini(cumulative_pays)

plt.figure(figsize=(8, 6))
im = plt.imshow(gini_matrix_gigworker_sched, origin='lower', aspect='auto', cmap='viridis',
                extent=[gig_worker_ratios[0], gig_worker_ratios[-1], sched_grid[0], sched_grid[-1]])
plt.colorbar(im, label="Gini Coefficient (Cumulative Pay)")
plt.xlabel("Ratio of Gigs to Workers")
plt.ylabel("Fraction of Workers with High Scheduling")
plt.title("Gini Coefficient Heatmap\n(Cumulative Pay vs. Gigs/Workers Ratio & High Scheduling)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_gini_gigworker_vs_highsched.png")
plt.close()

