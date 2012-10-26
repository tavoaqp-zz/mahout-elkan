package org.apache.mahout.clustering.elkan;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.Writable;
public class ElkanVectorWritable extends Configured implements Writable{

	private ElkanVector mappedVector;
	
	public ElkanVectorWritable()
	{
	}
	
	public ElkanVectorWritable(ElkanVector mappedVector)
	{
		setMappedVector(mappedVector);
	}
	
	public void setMappedVector(ElkanVector mappedVector)
	{
		this.mappedVector=mappedVector;
	}
	
	public ElkanVector get()
	{
		return this.mappedVector;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		mappedVector=new ElkanVector();
		mappedVector.read(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		mappedVector.write(out);		
	}

}
