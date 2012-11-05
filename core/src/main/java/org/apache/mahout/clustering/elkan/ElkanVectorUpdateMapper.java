package org.apache.mahout.clustering.elkan;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.common.iterator.sequencefile.ElkanVectorFilter;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class ElkanVectorUpdateMapper extends
		ElkanMapper<WritableComparable<?>, ElkanVectorWritable> {

	private Vector oldClusterDistances;

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {

		super.setup(context);

		Configuration conf = context.getConfiguration();
		String newPriorClustersPath = conf
				.get(ElkanIterator.NEW_PRIOR_PATH_KEY);

		List<Cluster> newModels = classifier.readNewModelsFromSeqFiles(conf,
				new Path(newPriorClustersPath));

		oldClusterDistances = new DenseVector(newModels.size());
		for (int i = 0; i < newModels.size(); i++) {
			Vector oldCenter = classifier.getModels().get(i).getCenter();
			Vector newCenter = newModels.get(i).getCenter();
			oldClusterDistances.setQuick(i,
					measure.distance(oldCenter, newCenter));
		}

	}

	@Override
	protected void map(WritableComparable<?> key, ElkanVectorWritable value,
			Context context) throws IOException, InterruptedException {
		ElkanVector elkanVector = value.get();
		ElkanSteps.updateVectorLimits(elkanVector, oldClusterDistances);
		ElkanSteps.updateVectorCentroid(elkanVector, classifier,
				medianDistanceClusters, clusterDistanceMatrix, measure);
		classifier.train(elkanVector.getClusterId(), elkanVector.getDelegate(), 1.0);
		m_multiOutputs.write(ElkanVectorFilter.ELKAN_VECTOR_PREFIX, key, new ElkanVectorWritable(elkanVector));
	}
}