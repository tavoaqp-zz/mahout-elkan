package org.apache.mahout.clustering.elkan;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.CIReducer;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ElkanStep2Job {

	public static int run(Path input, Path output, Configuration conf) {
		try {
			Job job = new Job(conf, "Calculate Clusters job.");

			job.setMapperClass(ElkanStep2Mapper.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(ClusterWritable.class);

			job.setReducerClass(CIReducer.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(ClusterWritable.class);

			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			FileOutputFormat.setOutputPath(job, output);

			job.setInputFormatClass(SequenceFileInputFormat.class);
			FileInputFormat.setInputPaths(job, input);

			job.setJarByClass(ElkanStep1Job.class);

			job.waitForCompletion(true);
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			return 1;
		}

		return 0;
	}

	public static class ElkanStep2Mapper
			extends
			Mapper<WritableComparable<?>, ElkanVectorWritable, IntWritable, ClusterWritable> {

		private static final Logger log = LoggerFactory
				.getLogger(ElkanStep2Mapper.class);

		private ClusterClassifier classifier;

		private ClusteringPolicy policy;

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			String priorClustersPath = conf.get(ClusterIterator.PRIOR_PATH_KEY);
			classifier = new ClusterClassifier();
			classifier.readFromSeqFiles(conf, new Path(priorClustersPath));
			policy = classifier.getPolicy();
			policy.update(classifier);
			super.setup(context);
		}

		@Override
		protected void map(WritableComparable<?> key,
				ElkanVectorWritable value, Context context) throws IOException,
				InterruptedException {
			ElkanVector vector = value.get();
			classifier.train(vector.getClusterId(), vector, 1.0);
		}

		@Override
		protected void cleanup(
				org.apache.hadoop.mapreduce.Mapper.Context context)
				throws IOException, InterruptedException {
			List<Cluster> clusters = classifier.getModels();
			ClusterWritable cw = new ClusterWritable();
			for (int index = 0; index < clusters.size(); index++) {
				cw.setValue(clusters.get(index));
				context.write(new IntWritable(index), cw);
			}
			super.cleanup(context);
		}
	}
}
