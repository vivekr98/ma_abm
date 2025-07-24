# --- USER CONFIGURATION ---
OUTPUT_DIR = "outputs2"
NUM_WORKERS = 100
NUM_GIGS = 50
STEPS = 100

PERCENT_DISCRIMINATORY_GIGS = 0.3  #Fraction of gigs with discriminatory_attitude > 0.5
PERCENT_HIGH_PERFORMANCE = 0.3    # Fraction of workers with performance_ability > 0.7
PERCENT_HIGH_SCHEDULING = 0.3    # Fraction of workers with scheduling > 0.7
PERCENT_SPOILED_IDENTITY = 0.3     # Fraction of workers with spoiled identity
SCHEDULING_THRESHOLD = 0.7         # Threshold for high scheduling (used in WorkerAgent.step)

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import random
from mesa import Agent, Model
from mesa.time import BaseScheduler
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns

# --- MODEL CODE ---

class GigAssignment:
    def __init__(self, gig_id, pay, discriminatory_attitude):
        self.gig_id = gig_id
        self.pay = pay
        self.discriminatory_attitude = discriminatory_attitude
        self.taken = False

    def rate_worker(self, worker):
        base_rating = random.uniform(3.0, 5.0)
        # Only penalize if gig is discriminatory (>0.5)
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
            # Pick job with highest pay
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

    def update_precarity(self, avg_choices):
        avg_pay = np.mean(self.pay_history[-5:]) if self.pay_history else 0
        limited_choices = 1 if (self.choices_history and self.choices_history[-1] < avg_choices) else 0
        low_pay_penalty = 1.0 if avg_pay < 0.5 else 0
        spell_penalty = 1.0 if self.spell_streak >= 3 else 0
        choice_penalty = limited_choices
        step_prec = (low_pay_penalty + spell_penalty + choice_penalty) / 3
        self.precarity_history.append(step_prec)
        self.precarity_index = np.mean(self.precarity_history)

    @property
    def rating(self):
        avg_rating = np.mean(self.rating_history) if self.rating_history else 5.0
        penalty = 1.0 if self.spell_streak >= 3 else 0.0
        return max(1.0, avg_rating - penalty)

class PeopleWorkModel(Model):
    def __init__(self, num_workers=NUM_WORKERS, num_gigs=NUM_GIGS):
        self.schedule = BaseScheduler(self)
        self.workers = []
        self.gigs = []

        # Initialize workers
        num_high_perf = int(PERCENT_HIGH_PERFORMANCE * num_workers)
        num_high_sched = int(PERCENT_HIGH_SCHEDULING * num_workers)
        num_spoiled = int(PERCENT_SPOILED_IDENTITY * num_workers)
        for i in range(num_workers):
            spoiled = random.random() < PERCENT_SPOILED_IDENTITY
            perf = random.uniform(0.8, 1.0) if random.random() < PERCENT_HIGH_PERFORMANCE else random.uniform(0, 0.7)
            scheduling = random.uniform(0.8, 1.0) if random.random() < PERCENT_HIGH_SCHEDULING else random.uniform(0, 0.7)
            worker = WorkerAgent(i, self, spoiled_identity=spoiled, performance_ability=perf, scheduling=scheduling)
            self.workers.append(worker)
            self.schedule.add(worker)

        # Initialize gigs (pay and discrimination correlated)
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

        # Sort workers by rating (highest first)
        sorted_workers = sorted(self.workers, key=lambda w: w.rating, reverse=True)

        # Each worker picks the best available gig, in order of rating
        available_gigs = [gig for gig in self.gigs if not gig.taken]
        for worker in sorted_workers:
            # Worker can see all gigs not yet taken
            choices = [g for g in available_gigs]
            worker.step(choices)
            # Remove taken gigs
            available_gigs = [g for g in available_gigs if not g.taken]

        # Calculate average choices offered last step
        avg_choices = np.mean([len([g for g in self.gigs if not g.taken]) for _ in self.workers])

        for worker in self.workers:
            worker.update_precarity(avg_choices)

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = np.array(array)
    if np.amin(array) < 0:
        array -= np.amin(array)  # Values cannot be negative
    array += 1e-10  # Avoid division by zero
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def lorenz_curve(array):
    """Return x and y values for Lorenz curve of a numpy array."""
    array = np.array(array)
    array = np.sort(array)
    cum_array = np.cumsum(array)
    total = cum_array[-1]
    x_lorenz = np.linspace(0, 1, len(array) + 1)
    y_lorenz = np.concatenate([[0], cum_array / total])
    return x_lorenz, y_lorenz

if __name__ == "__main__":
    model = PeopleWorkModel(num_workers=NUM_WORKERS, num_gigs=NUM_GIGS)
    for i in range(STEPS):
        model.step()

    # Normalize precarity index for each worker
    max_prec = max(max(w.precarity_history) for w in model.workers)
    for worker in model.workers:
        worker.normalized_precarity_history = [p / max_prec if max_prec > 0 else 0 for p in worker.precarity_history]

    # Visualization: colored lines
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        plt.plot(worker.normalized_precarity_history, color=color, linestyle=linestyle, alpha=0.7)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Spoiled Identity'),
        Line2D([0], [0], color='blue', lw=2, label='Non-Spoiled Identity'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='High Performance (>0.7)'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Low Performance (≤0.7)'),
    ]
    plt.xlabel("Step")
    plt.ylabel("Normalized Precarity Index")
    plt.title("Worker Normalized Precarity Index Over Time")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/normalized_precarity_index_over_time.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        plt.plot(worker.rating_history, color=color, linestyle=linestyle, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Worker Rating")
    plt.title("Worker Rating Over Time")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/worker_rating_over_time.png")
    plt.close()

    # --- Collect worker-level data ---
    worker_data = []
    for worker in model.workers:
        worker_data.append({
            "worker_id": worker.unique_id,
            "spoiled_identity": worker.spoiled_identity,
            "performance_ability": worker.performance_ability,
            "scheduling": worker.scheduling,
            "avg_precarity": np.mean(worker.precarity_history),
            "std_precarity": np.std(worker.precarity_history),
            "avg_rating": np.mean(worker.rating_history),
            "std_rating": np.std(worker.rating_history),
            "avg_pay": np.mean(worker.pay_history),
            "missed_jobs": sum([1 if pay == 0 else 0 for pay in worker.pay_history]),
            "total_steps": len(worker.pay_history),
        })
    df_workers = pd.DataFrame(worker_data)

    # --- Collect gig-level data ---
    gig_data = []
    for gig in model.gigs:
        gig_data.append({
            "gig_id": gig.gig_id,
            "pay": gig.pay,
            "discriminatory_attitude": gig.discriminatory_attitude,
        })
    df_gigs = pd.DataFrame(gig_data)

    # --- Model-level summary ---
    model_summary = {
        "num_workers": len(model.workers),
        "num_gigs": len(model.gigs),
        "percent_discriminatory": np.mean([gig.discriminatory_attitude > 0.7 for gig in model.gigs]),
        "percent_high_performance": np.mean([worker.performance_ability > 0.7 for worker in model.workers]),
        "percent_high_scheduling": np.mean([worker.scheduling > 0.7 for worker in model.workers]),
        "percent_spoiled": np.mean([worker.spoiled_identity for worker in model.workers]),
    }

    # --- Grouped summaries ---
    grouped = {}
    grouped['by_spoiled'] = df_workers.groupby("spoiled_identity")["avg_precarity"].describe()
    grouped['by_performance'] = df_workers.groupby(df_workers["performance_ability"] > 0.7)["avg_precarity"].describe()
    grouped['by_scheduling'] = df_workers.groupby(df_workers["scheduling"] > 0.7)["avg_precarity"].describe()
    grouped['joint'] = df_workers.groupby(
        ["spoiled_identity", df_workers["performance_ability"] > 0.7, df_workers["scheduling"] > 0.7]
    )["avg_precarity"].describe()

    # --- Save worker-level data ---
    df_workers.to_csv(f"{OUTPUT_DIR}/worker_analysis.csv", index=False)
    # --- Save gig-level data ---
    df_gigs.to_csv(f"{OUTPUT_DIR}/gig_analysis.csv", index=False)
    # --- Save model summary ---
    pd.DataFrame([model_summary]).to_csv(f"{OUTPUT_DIR}/model_summary.csv", index=False)
    # --- Save grouped summaries ---
    for key, df_group in grouped.items():
        df_group.to_csv(f"{OUTPUT_DIR}/precarity_summary_{key}.csv")

    # --- Optionally, save a joint table with all three labels ---
    joint_table = df_workers.copy()
    joint_table["high_performance"] = joint_table["performance_ability"] > 0.7
    joint_table["high_scheduling"] = joint_table["scheduling"] > 0.7
    joint_table.to_csv(f"{OUTPUT_DIR}/worker_joint_table.csv", index=False)

    # --- Cumulative Average Graphs ---

    # Cumulative average precarity
    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        running_avg = np.cumsum(worker.precarity_history) / (np.arange(len(worker.precarity_history)) + 1)
        plt.plot(running_avg, color=color, linestyle=linestyle, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Average Precarity Index")
    plt.title("Worker Cumulative Average Precarity Index Over Time")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_avg_precarity_index_over_time.png")
    plt.close()

    # Cumulative jobs taken
    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        jobs_taken = [1 if pay > 0 else 0 for pay in worker.pay_history]
        cumulative_jobs_taken = np.cumsum(jobs_taken)
        plt.plot(cumulative_jobs_taken, color=color, linestyle=linestyle, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Jobs Taken")
    plt.title("Worker Cumulative Jobs Taken Over Time")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_jobs_taken_over_time.png")
    plt.close()

    # Cumulative average jobs presented
    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        running_avg_choices = np.cumsum(worker.choices_history) / (np.arange(len(worker.choices_history)) + 1)
        plt.plot(running_avg_choices, color=color, linestyle=linestyle, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Avg. Jobs Presented")
    plt.title("Worker Cumulative Avg. Jobs Presented Over Time")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_avg_jobs_presented_over_time.png")
    plt.close()

    # Cumulative average pay
    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        running_avg_pay = np.cumsum(worker.pay_history) / (np.arange(len(worker.pay_history)) + 1)
        plt.plot(running_avg_pay, color=color, linestyle=linestyle, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Avg. Pay")
    plt.title("Worker Cumulative Avg. Pay Over Time")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_avg_pay_over_time.png")
    plt.close()


    # --- Cumulative Average Rating ---
    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "red" if worker.spoiled_identity else "blue"
        linestyle = "-" if worker.performance_ability > 0.7 else "--"
        running_avg_rating = np.cumsum(worker.rating_history) / (np.arange(len(worker.rating_history)) + 1)
        plt.plot(running_avg_rating, color=color, linestyle=linestyle, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Average Worker Rating")
    plt.title("Worker Cumulative Average Rating Over Time (by Identity & Performance)")
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_avg_worker_rating_by_identity_performance.png")
    plt.close()




    # Cumulative average rating by flexibility
    plt.figure(figsize=(10, 6))
    for worker in model.workers:
        color = "green" if worker.scheduling > SCHEDULING_THRESHOLD else "orange"
        running_avg_rating = np.cumsum(worker.rating_history) / (np.arange(len(worker.rating_history)) + 1)
        plt.plot(running_avg_rating, color=color, alpha=0.7)
    legend_elements_flex = [
        Line2D([0], [0], color='green', lw=2, label='High Flexibility (scheduling > 0.7)'),
        Line2D([0], [0], color='orange', lw=2, label='Low Flexibility (scheduling ≤ 0.7)'),
    ]
    plt.xlabel("Step")
    plt.ylabel("Cumulative Average Worker Rating")
    plt.title("Worker Cumulative Average Rating Over Time (by Scheduling Flexibility)")
    plt.legend(handles=legend_elements_flex, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_avg_worker_rating_by_scheduling.png")
    plt.close()

    # --- Histograms with KDE Smoothing ---
    import seaborn as sns

    all_ratings = [float(rating) for worker in model.workers for rating in worker.rating_history]
    spoiled_ratings = [float(rating) for worker in model.workers if worker.spoiled_identity for rating in worker.rating_history]
    nonspoiled_ratings = [float(rating) for worker in model.workers if not worker.spoiled_identity for rating in worker.rating_history]
    high_perf_ratings = [float(rating) for worker in model.workers if worker.performance_ability > 0.7 for rating in worker.rating_history]
    low_perf_ratings = [float(rating) for worker in model.workers if worker.performance_ability <= 0.7 for rating in worker.rating_history]

    # All ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(all_ratings, bins=30, color='gray', kde=True)
    plt.xlabel("Worker Rating")
    plt.ylabel("Frequency")
    plt.title("Histogram of All Worker Ratings with Smoothing")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/histogram_worker_ratings_smoothed.png")
    plt.close()

    # By performance
    plt.figure(figsize=(10, 6))
    sns.histplot(high_perf_ratings, bins=30, color='black', kde=True, label='High Performance (>0.7)', alpha=0.5)
    sns.histplot(low_perf_ratings, bins=30, color='gray', kde=True, label='Low Performance (≤0.7)', alpha=0.5)
    plt.xlabel("Worker Rating")
    plt.ylabel("Frequency")
    plt.title("Histogram of Worker Ratings by Performance Ability with Smoothing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/histogram_worker_ratings_by_performance.png")
    plt.close()

    # By spoiled identity
    plt.figure(figsize=(10, 6))
    sns.histplot(spoiled_ratings, bins=30, color='red', kde=True, label='Spoiled Identity', alpha=0.6)
    sns.histplot(nonspoiled_ratings, bins=30, color='blue', kde=True, label='Non-Spoiled Identity', alpha=0.6)
    plt.xlabel("Worker Rating")
    plt.ylabel("Frequency")
    plt.title("Histogram of Worker Ratings by Spoiled Identity with Smoothing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/histogram_worker_ratings_by_identity_smoothed.png")
    plt.close()


    spoiled = np.array([w.spoiled_identity for w in model.workers])
    high_perf = np.array([w.performance_ability > 0.7 for w in model.workers])
    cumulative_pays = np.array([np.sum(w.pay_history) for w in model.workers])

    group_both = spoiled & high_perf
    group_spoiled_only = spoiled & ~high_perf
    group_high_perf_only = ~spoiled & high_perf
    group_neither = ~spoiled & ~high_perf

    group_labels = [
        ("Both Spoiled & High Perf", group_both, "purple"),
        ("Spoiled Only", group_spoiled_only, "red"),
        ("High Perf Only", group_high_perf_only, "blue"),
        ("Neither", group_neither, "gray"),
]

    xmin = cumulative_pays.min() - 1
    xmax = cumulative_pays.max() + 1

    # Get all densities to find ymax
    all_densities = []
    for data in [
        cumulative_pays[group_both], cumulative_pays[group_spoiled_only],
        cumulative_pays[group_high_perf_only], cumulative_pays[group_neither]
    ]:
        if len(data) > 1:
            kde = sns.kdeplot(data).get_lines()[0].get_data()
            all_densities.append(kde[1])
            plt.clf()  # Clear the plot after getting the density

    if all_densities:
        ymax = max([d.max() for d in all_densities]) * 1.05
    else:
        ymax = 1

    plt.figure(figsize=(18, 5))

    # 1. Both spoiled & high perf vs neither
    plt.subplot(1, 3, 1)
    if cumulative_pays[group_both].size > 1:
        sns.kdeplot(cumulative_pays[group_both], color="purple", label="Both Spoiled & High Perf", fill=True, alpha=0.4, linewidth=2)
    if cumulative_pays[group_neither].size > 1:
        sns.kdeplot(cumulative_pays[group_neither], color="gray", label="Neither", fill=True, alpha=0.4, linewidth=2)
    plt.title("Both Spoiled & High Perf vs Neither")
    plt.xlabel("Worker Cumulative Pay")
    plt.ylabel("Density")
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.legend()

    # 2. Spoiled only vs neither
    plt.subplot(1, 3, 2)
    if cumulative_pays[group_spoiled_only].size > 1:
        sns.kdeplot(cumulative_pays[group_spoiled_only], color="red", label="Spoiled Only", fill=True, alpha=0.4, linewidth=2)
    if cumulative_pays[group_neither].size > 1:
        sns.kdeplot(cumulative_pays[group_neither], color="gray", label="Neither", fill=True, alpha=0.4, linewidth=2)
    plt.title("Spoiled Only vs Neither")
    plt.xlabel("Worker Cumulative Pay")
    plt.ylabel("Density")
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.legend()

    # 3. High perf only vs neither
    plt.subplot(1, 3, 3)
    if cumulative_pays[group_high_perf_only].size > 1:
        sns.kdeplot(cumulative_pays[group_high_perf_only], color="blue", label="High Perf Only", fill=True, alpha=0.4, linewidth=2)
    if cumulative_pays[group_neither].size > 1:
        sns.kdeplot(cumulative_pays[group_neither], color="gray", label="Neither", fill=True, alpha=0.4, linewidth=2)
    plt.title("High Perf Only vs Neither")
    plt.xlabel("Worker Cumulative Pay")
    plt.ylabel("Density")
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kde_cumulative_pay_group_vs_neither.png")
    plt.close()

    # --- Gig Discrimination vs Worker Rating ---
    ratings_data = []
    for gig in model.gigs:
        for worker in model.workers:
            for step, pay in enumerate(worker.pay_history):
                if pay == gig.pay and gig.discriminatory_attitude > 0.5:
                    ratings_data.append({
                        "gig_id": gig.gig_id,
                        "discriminatory_attitude": gig.discriminatory_attitude,
                        "worker_rating": worker.rating_history[step],
                        "spoiled_identity": worker.spoiled_identity
                    })
    df_ratings = pd.DataFrame(ratings_data)
    plt.figure(figsize=(8, 6))
    for spoiled, group in df_ratings.groupby("spoiled_identity"):
        plt.scatter(group["discriminatory_attitude"], group["worker_rating"],
                    label=f"Spoiled: {spoiled}", alpha=0.7)
    plt.xlabel("Gig Discriminatory Attitude")
    plt.ylabel("Worker Rating Received")
    plt.title("Ratings Given by Discriminatory Employers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gig_discrimination_vs_worker_rating.png")
    plt.close()

    # --- Worker Pay Trajectories Colored by Rating ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for worker in model.workers:
        steps = np.arange(len(worker.pay_history))
        pay = worker.pay_history
        ratings = worker.rating_history
        norm_ratings = (np.array(ratings) - 1) / 4
        colors = cm.viridis(norm_ratings)
        ax.scatter(steps, pay, c=colors, label=None, alpha=0.7, s=30)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pay")
    ax.set_title("Worker Pay Trajectories Colored by Rating")
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Worker Rating")
    yticks = ax.get_yticks()
    yticklabels = [("no gig held" if abs(y) < 1e-6 else f"{y:.1f}") for y in yticks]
    ax.set_yticklabels(yticklabels)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/worker_pay_trajectories_colored_by_rating.png")
    plt.close()

    # Calculate cumulative pay for each worker
    cumulative_pays = [np.sum(worker.pay_history) for worker in model.workers]

    # Calculate Gini coefficient
    gini_pay = gini(cumulative_pays)
    print(f"Gini coefficient for cumulative pay: {gini_pay:.3f}")

    # Visualize cumulative pay distribution
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



    def gini_2(array):
        """Calculate the Gini coefficient of a numpy array."""
        array = np.array(array, dtype=float)  # Ensure float dtype
        if np.amin(array) < 0:
            array -= np.amin(array)  # Values cannot be negative
        array += 1e-10  # Avoid division by zero
        array = np.sort(array)
        n = array.size
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    jobs_presented = [np.sum(worker.choices_history) for worker in model.workers]

# Gini coefficient
    gini_jobs = gini_2(jobs_presented)
    print(f"Gini coefficient for jobs presented: {gini_jobs:.3f}")

    # Distribution graph
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(jobs_presented)), sorted(jobs_presented), color='teal', alpha=0.7)
    plt.xlabel("Worker (sorted by jobs presented)")
    plt.ylabel("Total Jobs Presented")
    plt.title(f"Jobs Presented Distribution (Gini: {gini_jobs:.3f})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/jobs_presented_distribution_gini.png")
    plt.close()

    # Lorenz curve
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