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
#include <linux/kobject.h>
#include <linux/sysfs.h>

#define DEFAULT_MIN_CPUS 1
#define DEFAULT_MAX_CPUS NR_CPUS


#define NR_FSHIFT_EXP	3
#define NR_FSHIFT	(1 << NR_FSHIFT_EXP)

/* avg run threads * 8 (e.g., 11 = 1.375 threads) */
static unsigned int thresholds_default[] = {
	10, 18, 20, UINT_MAX
};

static unsigned int thresholds_select = 0;

typedef enum {
	DISABLED,
	RUNNING,
} RUNNABLES_STATE;

static struct work_struct runnables_work;
static struct timer_list runnables_timer;

/* configurable parameters */
static RUNNABLES_STATE runnables_state;
static unsigned int sample_rate = 20;		/* msec */
module_param_named(sample_rate, sample_rate, uint, 0644);

static unsigned int nr_run_last;
static unsigned int nr_run_hysteresis = 2;		/* 1 / 2 thread */
module_param_named(nr_run_hysteresis, nr_run_hysteresis, uint, 0644);
static unsigned int default_threshold_level = 4;	/* 1 / 4 thread */
static unsigned int nr_run_thresholds[NR_CPUS];
static unsigned int max_cpus = DEFAULT_MAX_CPUS;
static unsigned int min_cpus = DEFAULT_MIN_CPUS;

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

	if (nr_run < min_cpus || nr_run > max_cpus)
		return 0;

	if ((nr_cpus > max_cpus || nr_run < nr_cpus) && nr_cpus >= min_cpus)
		return -1;

	if ((nr_cpus < min_cpus || nr_run > nr_cpus) && nr_cpus <= max_cpus)
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

	for (nr_run = 1; nr_run < ARRAY_SIZE(thresholds_default); nr_run++) {
		unsigned int nr_threshold = thresholds_default[nr_run - 1];
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
	struct cpumask mask;

	if (runnables_state != RUNNING)
		return;

	action = get_action(nr_run_last);
	if (action > 0) {
			/*get cpus that are offline (invert of cpu_online_mask to mask)*/
			cpumask_complement(&mask, cpu_online_mask);
			/* take one of them (random) and bring it online*/
			cpu = cpumask_any(&mask);
			if (cpu < nr_cpu_ids)
				cpu_up(cpu);
	} else if (action < 0) {
		cpu = get_lightest_loaded_cpu_n();
		if (cpu < nr_cpu_ids)
			cpu_down(cpu);
	}
}

/*************************************sysfs*******************************************/
static ssize_t runnables_on_show (struct kobject *kobj, struct kobj_attribute *attr, 
					char *buf)
{
	return sprintf(buf, "%u\n", runnables_state);
}
static ssize_t runnables_on_store(struct kobject *kobj, struct kobj_attribute *attr, 
					const char *buf, size_t count)
{
	int ret;
	unsigned int val;
	ret = sscanf(buf, "%u", &val);
	if (ret != 1 || val < 0 || val > 1)
		return -EINVAL;
	
	if (val == 1){
		mutex_lock(&runnables_lock);
		runnables_state = RUNNING;
		mutex_unlock(&runnables_lock);

		runnables_avg_sampler(0);
	} else {
		mutex_lock(&runnables_lock);
		cancel_work_sync(&runnables_work);
		runnables_state = DISABLED;
		mutex_unlock(&runnables_lock);
	}

	return count;
}

static ssize_t min_cpus_show (struct kobject *kobj, struct kobj_attribute *attr, 
					char *buf)
{
	return sprintf(buf, "%u\n", min_cpus);
}

static ssize_t min_cpus_store(struct kobject *kobj, struct kobj_attribute *attr, 
					const char *buf, size_t count)
{
	int ret;
	unsigned int val;
	ret = sscanf(buf, "%u", &val);
	if (ret != 1 || val < 1 || val > 4)
		return -EINVAL;
	
	if (val > max_cpus)
		val = max_cpus;

	min_cpus = val;

	return count;
}

static ssize_t max_cpus_show (struct kobject *kobj, struct kobj_attribute *attr, 
					char *buf)
{
	return sprintf(buf, "%u\n", max_cpus);
}

static ssize_t max_cpus_store(struct kobject *kobj, struct kobj_attribute *attr, 
					const char *buf, size_t count)
{
	int ret;
	unsigned int val;
	ret = sscanf(buf, "%u", &val);
	if (ret != 1 || val < 1 || val > 4)
		return -EINVAL;
	
	if (val < min_cpus)
		val = min_cpus;

	max_cpus = val;

	return count;
}

static ssize_t thresholds_presets_show (struct kobject *kobj, struct kobj_attribute *attr, 
					char *buf)
{
	return sprintf(buf, "%u\n", thresholds_select);
}

static ssize_t thresholds_presets_store(struct kobject *kobj, struct kobj_attribute *attr, 
					const char *buf, size_t count)
{
	int ret;
	unsigned int val;
	ret = sscanf(buf, "%u", &val);

	if (ret != 1)
		return -EINVAL;

	switch (val){
		case 0:
			thresholds_default[0] = 10;
			thresholds_default[1] = 18;
			thresholds_default[2] = 20;
			break;
		case 1:
			thresholds_default[0] = 8;
			thresholds_default[1] = 16;
			thresholds_default[2] = 18;
			break;
		case 2:
			thresholds_default[0] = 14;
			thresholds_default[1] = 26;
			thresholds_default[2] = 28;
			break;
		default:
			return -EINVAL;
	}

	thresholds_select = val;

	pr_info("%s: Thresholds are now = %d, %d, %d.\n", KBUILD_MODNAME, thresholds_default[0], 
		thresholds_default[1], thresholds_default[2]);

	return count;
}

static ssize_t thresholds_show(struct kobject *kobj, struct kobj_attribute *attr, 
					char *buf)
{
	char *out = buf;
	int i = 0;
	for(i = 0; i < 3; i++)
 		out += sprintf(out, "%u ",thresholds_default[i]);

	return out - buf;
}

static ssize_t thresholds_store(struct kobject *kobj, struct kobj_attribute *attr, 
					const char *buf, size_t count)
{
	int ret, i = 0;
	unsigned int val[3];
	ret = sscanf(buf, "%u %u %u", &val[0], &val[1], &val[2]);
	if (ret != 3)
		return -EINVAL;
	
	for(i = 0; i < 3; i++)
		thresholds_default[i] = val[i];

	pr_info("%s: Thresholds are now = %d, %d, %d.\n", KBUILD_MODNAME, thresholds_default[0], 
		thresholds_default[1], thresholds_default[2]);

	return count;
}

static struct kobj_attribute runnables_on_attribute =
	__ATTR(runnables_on, 0644, runnables_on_show, runnables_on_store);
static struct kobj_attribute min_cpus_attribute =
	__ATTR(min_cpus, 0644, min_cpus_show, min_cpus_store);
static struct kobj_attribute max_cpus_attribute =
	__ATTR(max_cpus, 0644, max_cpus_show, max_cpus_store);
static struct kobj_attribute thresholds_presets_attribute =
	__ATTR(thresholds_presets, 0644, thresholds_presets_show, thresholds_presets_store);
static struct kobj_attribute thresholds_attribute =
	__ATTR(thresholds, 0644, thresholds_show, thresholds_store);

static struct attribute *attrs[] = {
	&runnables_on_attribute.attr,
	&min_cpus_attribute.attr,
	&max_cpus_attribute.attr,
	&thresholds_presets_attribute.attr,
	&thresholds_attribute.attr,
	NULL,
};
/*************************************sysfs end****************************************/

static struct attribute_group attr_group = {
	.attrs = attrs,
};

static int __init runnables_hotplug_init(void)
{	

	int i, ret = 0;
	struct kobject *module_kobj;

	printk(KERN_INFO "RUNNABLES init!\n");

	module_kobj = kset_find_obj(module_kset, KBUILD_MODNAME);
	if (!module_kobj) {
		pr_err("%s: cannot find kobject for module %s\n", __func__, KBUILD_MODNAME);
		ret = -ENOENT;
		goto failed;
	}
	
	
	ret = sysfs_create_group(module_kobj, &attr_group);
	if (ret) {
		pr_err("%s: cannot create kobject attribute group\n", __func__);
		goto failed;
	}

	INIT_WORK(&runnables_work, runnables_work_func);

	init_timer(&runnables_timer);
	runnables_timer.function = runnables_avg_sampler;

	for(i = 0; i < ARRAY_SIZE(nr_run_thresholds); ++i) {
		if (i < ARRAY_SIZE(thresholds_default))
			nr_run_thresholds[i] = thresholds_default[i];
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
failed:
 	kobject_put(module_kobj);
	
	return ret;
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
