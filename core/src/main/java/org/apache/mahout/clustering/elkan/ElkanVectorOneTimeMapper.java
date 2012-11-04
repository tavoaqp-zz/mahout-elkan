package org.apache.mahout.clustering.elkan;

import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.common.iterator.sequencefile.ElkanVectorFilter;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class ElkanVectorOneTimeMapper extends
		ElkanMapper<WritableComparable<?>, VectorWritable> {

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {		
		super.setup(context);
	}

	@Override
	protected void map(WritableComparable<?> key, VectorWritable value,
			Context context) throws IOException, InterruptedException {
		Vector originVector = value.get();
		int clusterIndex = 0;
		Vector probabilities = new DenseVector(classifier.getModels().size());
		for (Cluster model : classifier.getModels()) {
			probabilities.setQuick(clusterIndex,
					measure.distance(originVector, model.getCenter()));
			clusterIndex++;
		}

		ElkanVector elkanVector = new ElkanVector();
		elkanVector.setDelegate(originVector);
		elkanVector.setLowerLimits(probabilities);
		elkanVector.setClusterId(probabilities.minValueIndex());
		elkanVector.setCalculateDistance(true);
		elkanVector.setUpperLimit(probabilities.minValue());
		ElkanSteps.updateVectorCentroid(elkanVector, classifier,
				medianDistanceClusters, clusterDistanceMatrix, measure);
		classifier.train(elkanVector.getClusterId(), elkanVector.getDelegate(), 1.0);
		m_multiOutputs.write(ElkanVectorFilter.ELKAN_VECTOR_PREFIX, key, new ElkanVectorWritable(elkanVector));
	}

}