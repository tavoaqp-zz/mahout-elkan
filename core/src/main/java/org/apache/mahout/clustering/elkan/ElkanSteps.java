package org.apache.mahout.clustering.elkan;

import java.util.List;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenHashMap;

public class ElkanSteps {

	public static void updateVectorCentroid(ElkanVector originVector,
			ClusterClassifier classifier, Vector halfClusterDistances,
			ElkanClusterDistancesCache clusterDistanceMatrix,
			DistanceMeasure measure) {
		List<Cluster> models = classifier.getModels();
		if (originVector.getUpperLimit() > halfClusterDistances
				.get(originVector.getClusterId())) {
			int currClusterId = 0;
			for (Cluster model : models) {
				Integer originClusterId = Integer.valueOf(originVector
						.getClusterId());
				if (currClusterId != originClusterId) {
					double lowerLimit = originVector.getLowerLimits().get(currClusterId);
					double upperLimit = originVector.getUpperLimit();
					Vector clusterVector = models.get(
							originVector.getClusterId()).getCenter();
					Vector clusterDistances=clusterDistanceMatrix.get(currClusterId); 
					double clusterDistance = clusterDistances.getQuick(originClusterId);

					double curDist = 0;
					if (upperLimit > lowerLimit
							&& upperLimit > clusterDistance / 2) {

						if (originVector.isCalculateDistance()) {
							curDist = measure.distance(originVector,
									clusterVector);
							originVector.getLowerLimits().setQuick(currClusterId, curDist);
							originVector.setCalculateDistance(false);
						} else {
							curDist = originVector.getUpperLimit();
						}
					}

					if (curDist > lowerLimit || curDist > clusterDistance / 2) {
						double newDist = measure.distance(model.getCenter(),
								originVector);
						originVector.getLowerLimits().setQuick(currClusterId, newDist);
						if (newDist < curDist) {
							originVector.setClusterId(currClusterId);
						}
					}
				}
				currClusterId++;
			}

		}
	}
	
	public static void updateVectorLimits(ElkanVector vector, Vector oldClusterDistances) {
		for(int i=0;i<oldClusterDistances.size();i++)
		{
			double clusterDist=oldClusterDistances.getQuick(i);
			double max=Math.max(vector.getLowerLimits().get(i)-clusterDist, 0);
			vector.getLowerLimits().setQuick(i, max);
		}
		double centerDist=oldClusterDistances.getQuick(vector.getClusterId());
		vector.setUpperLimit(vector.getUpperLimit()+centerDist);
		vector.setCalculateDistance(true);
	}
}
