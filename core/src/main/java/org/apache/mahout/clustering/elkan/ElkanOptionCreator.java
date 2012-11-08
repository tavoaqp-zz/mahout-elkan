package org.apache.mahout.clustering.elkan;

import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

public class ElkanOptionCreator {
	
	public static final String GENERATE_CLUSTERS_OPTION = "genClusters";
	
	public static DefaultOptionBuilder numClustersOption() {
	    return new DefaultOptionBuilder()
	        .withLongName(DefaultOptionCreator.NUM_CLUSTERS_OPTION)
	        .withRequired(true)
	        .withArgument(
	            new ArgumentBuilder().withName("k").withMinimum(1).withMaximum(1)
	                .create()).withDescription("The number of clusters to create")
	        .withShortName("k");
	  }
	
	public static DefaultOptionBuilder generateClusters() {
	    return new DefaultOptionBuilder()
	        .withLongName(GENERATE_CLUSTERS_OPTION)
	        .withRequired(false)
	        .withArgument(
	            new ArgumentBuilder().withName("genClusters").withMinimum(1).withMaximum(1)
	                .create()).withDescription("The number of clusters to create");
	  }

}
