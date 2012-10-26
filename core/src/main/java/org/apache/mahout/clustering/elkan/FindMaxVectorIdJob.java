package org.apache.mahout.clustering.elkan;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.iterator.ClusterWritable;

public class FindMaxVectorIdJob {
	
	public static int run(Path input, Path output, Configuration conf) {
		try {
			Job job = new Job(conf, "Find Max Vector Id job.");
			
			job.setMapperClass(FindMaxVectorIdMapper.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(IntWritable.class);

			job.setReducerClass(FindMaxVectorIdReducer.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(IntWritable.class);

			job.setCombinerClass(FindMaxVectorIdCombiner.class);
			
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			FileOutputFormat.setOutputPath(job, output);

			job.setInputFormatClass(SequenceFileInputFormat.class);
			FileInputFormat.setInputPaths(job, input);

			job.setJarByClass(FindMaxVectorIdJob.class);

			job.waitForCompletion(true);
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			return 1;
		}

		return 0;
	}

	
	public static Integer findMaxValue(Iterator<IntWritable> it)
	{
		Integer maxValue=-1;
		while (it.hasNext())
		{
			Integer curValue=it.next().get();
			maxValue=Math.max(maxValue, curValue);
		}
		return maxValue;
	}
	
	static class FindMaxVectorIdMapper extends Mapper<IntWritable,ElkanVectorWritable, IntWritable, IntWritable> 
	{		
		protected void map(IntWritable key, ElkanVectorWritable value, Context context) throws IOException, InterruptedException 
		{
			IntWritable newKey=new IntWritable();
			newKey.set(key.get() % 2);
			context.write(newKey,key );
		}		
	}
	
	static class FindMaxVectorIdCombiner extends Reducer<IntWritable,IntWritable, IntWritable, IntWritable>
	{
		IntWritable reducerId=new IntWritable(0);
		
		protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException 
		{
			Integer maxValue=findMaxValue(values.iterator());
			IntWritable maxWritable=new IntWritable();
			maxWritable.set(maxValue);
			context.write(reducerId, maxWritable);
		}
	}
	
	static class FindMaxVectorIdReducer extends Reducer<IntWritable,IntWritable, IntWritable, IntWritable> 
	{
		protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException 
		{
			Integer maxValue=findMaxValue(values.iterator());
			IntWritable maxWritable=new IntWritable();
			maxWritable.set(maxValue);
			context.write(maxWritable, maxWritable);
		}
	}
	
}
