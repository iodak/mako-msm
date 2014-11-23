/*
 * Copyright (c) 2012 NVIDIA CORPORATION.  All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; version 2 of the License.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */
/*Modded and ported to mako kernel by iodak o93.ivan@gmail.com*/
 

#include <linux/kernel.h>
#include <linux/cpumask.h>
#include <linux/module.h>
#include <linux/jiffies.h>
#include <linux/slab.h>
#include <linux/cpu.h>
#include <linux/sched.h>

#define DEFAULT_MIN_CPUS 1
#define DEFAULT_MAX_CPUS NR_CPUS

typedef enum {
	DISABLED,
	IDLE,
	RUNNING,
} RUNNABLES_STATE;

static struct work_struct runnables_work;
static struct timer_list runnables_timer;

static RUNNABLES_STATE runnables_state;
/* configurable parameters */
static unsigned int sample_rate = 20;		/* msec */
module_param(sample_rate, uint, 0644);

#define NR_FSHIFT_EXP	3
#define NR_FSHIFT	(1 << NR_FSHIFT_EXP)
/* avg run threads * 8 (e.g., 11 = 1.375 threads) */
static unsigned int default_thresholds[] = {
	10, 18, 20, UINT_MAX
};

static unsigned int nr_run_last;
static unsigned int nr_run_hysteresis = 2;		/* 1 / 2 thread */
module_param(nr_run_hysteresis, uint, 0644);

static unsigned int default_threshold_level = 4;	/* 1 / 4 thread */
static unsigned int nr_run_thresholds[NR_CPUS];
static wait_queue_head_t wait_cpu;

DEFINE_MUTEX(runnables_lock);

struct runnables_avg_sample {
	u64 previous_integral;
	unsigned int avg;
	bool integral_sampled;
	u64 prev_timestamp;
};

static DEFINE_PER_CPU(struct runnables_avg_sample, avg_nr_sample);

/* EXP = alpha in the exponential moving average.
 * Alpha = e ^ (-sample_rate / window_size) * FIXED_1
 * Calculated for sample_rate of 20ms, window size of 100ms
 */
#define EXP    1677

static unsigned int get_avg_nr_runnables(void)
{
	unsigned int i, sum = 0;
	static unsigned int avg;
	struct runnables_avg_sample *sample;
	u64 integral, old_integral, delta_integral, delta_time, cur_time;

	for_each_online_cpu(i) {
		sample = &per_cpu(avg_nr_sample, i);
		integral = nr_running_integral(i);
		old_integral = sample->previous_integral;
		sample->previous_integral = integral;
		cur_time = ktime_to_ns(ktime_get());
		delta_time = cur_time - sample->prev_timestamp;
		sample->prev_timestamp = cur_time;

		if (!sample->integral_sampled) {
			sample->integral_sampled = true;
			/* First sample to initialize prev_integral, skip
			 * avg calculation
			 */
			continue;
		}

		if (integral < old_integral) {
			/* Overflow */
			delta_integral = (ULLONG_MAX - old_integral) + integral;
		} else {
			delta_integral = integral - old_integral;
		}

		/* Calculate average for the previous sample window */
		do_div(delta_integral, delta_time);
		sample->avg = delta_integral;
		sum += sample->avg;
	}

	/* Exponential moving average
	 * Avgn = Avgn-1 * alpha + new_avg * (1 - alpha)
	 */
	avg *= EXP;
	avg += sum * (FIXED_1 - EXP);
	avg >>= FSHIFT;

	return avg;
}

static int get_action(unsigned int nr_run)
{
	unsigned int nr_cpus = num_online_cpus();
	int max_cpus = DEFAULT_MAX_CPUS;
	int min_cpus = DEFAULT_MIN_CPUS;

	if ((nr_cpus > max_cpus || nr_run < nr_cpus) && nr_cpus >= min_cpus)
		return -1;

	if (nr_cpus < min_cpus || nr_run > nr_cpus)
		return 1;

	return 0;
}

static void runnables_avg_sampler(unsigned long data)
{
	unsigned int nr_run, avg_nr_run;
	int action;

	rmb();
	if (runnables_state != RUNNING)
		return;

	avg_nr_run = get_avg_nr_runnables();
	mod_timer(&runnables_timer, jiffies + msecs_to_jiffies(sample_rate));

	for (nr_run = 1; nr_run < ARRAY_SIZE(nr_run_thresholds); nr_run++) {
		unsigned int nr_threshold = nr_run_thresholds[nr_run - 1];
		if (nr_run_last <= nr_run)
			nr_threshold += NR_FSHIFT / nr_run_hysteresis;
		if (avg_nr_run <= (nr_threshold << (FSHIFT - NR_FSHIFT_EXP)))
			break;
	}

	nr_run_last = nr_run;

	action = get_action(nr_run);
	if (action != 0) {
		wmb();
		schedule_work(&runnables_work);
	}
}

static unsigned int get_lightest_loaded_cpu_n(void)
{
	unsigned long min_avg_runnables = ULONG_MAX;
	unsigned int cpu = nr_cpu_ids;
	int i;

	for_each_online_cpu(i) {
		struct runnables_avg_sample *s = &per_cpu(avg_nr_sample, i);
		unsigned int nr_runnables = s->avg;
		if (i > 0 && min_avg_runnables > nr_runnables) {
			cpu = i;
			min_avg_runnables = nr_runnables;
		}
	}

	return cpu;
}

static void __ref runnables_work_func(struct work_struct *work)
{
	unsigned int cpu;
	int action;

	if (runnables_state != RUNNING)
		return;

	action = get_action(nr_run_last);
	if (action > 0) {
		for_each_cpu_not(cpu, cpu_online_mask)
			cpu_up(cpu);
	} else if (action < 0) {
		cpu = get_lightest_loaded_cpu_n();
		if (cpu < nr_cpu_ids)
			cpu_down(cpu);
	}
}

static int __init runnables_hotplug_init(void)
{	

	int i;

	printk(KERN_INFO "RUNABLESS init!\n");

	init_waitqueue_head(&wait_cpu);

	INIT_WORK(&runnables_work, runnables_work_func);

	init_timer(&runnables_timer);
	runnables_timer.function = runnables_avg_sampler;

	for(i = 0; i < ARRAY_SIZE(nr_run_thresholds); ++i) {
		if (i < ARRAY_SIZE(default_thresholds))
			nr_run_thresholds[i] = default_thresholds[i];
		else if (i == (ARRAY_SIZE(nr_run_thresholds) - 1))
			nr_run_thresholds[i] = UINT_MAX;
		else
			nr_run_thresholds[i] = i + 1 +
				NR_FSHIFT / default_threshold_level;
	}

	mutex_lock(&runnables_lock);
	runnables_state = RUNNING;
	mutex_unlock(&runnables_lock);

	runnables_avg_sampler(0);

	return 0;
}

void __exit runnables_hotplug_exit(void)
{
	mutex_lock(&runnables_lock);

	runnables_state = DISABLED;
	del_timer_sync(&runnables_timer);
	cancel_work_sync(&runnables_work);
	mutex_unlock(&runnables_lock);

	printk(KERN_INFO "Cleaning RUNNABLES\n");
}

late_initcall(runnables_hotplug_init);
module_exit(runnables_hotplug_exit);
MODULE_LICENSE("GPL");
