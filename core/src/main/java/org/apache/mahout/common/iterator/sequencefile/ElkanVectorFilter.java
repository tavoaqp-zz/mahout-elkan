package org.apache.mahout.common.iterator.sequencefile;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;

/**
 * This class filters out files that do not contain vectors but returns also the parent folder
 * containing those vectors.
 * @author gustavo
 *
 */
public class ElkanVectorFilter implements PathFilter {

	public static String ELKAN_VECTOR_PREFIX = "elkanVectors";

	@Override
	public boolean accept(Path path) {
		String name = path.getName();
		return name.contains("elkanVectors")
				|| (!name.contains("_SUCCESS") && !name.contains("_policy")
						&& !name.contains("part") && !name.endsWith(".crc")
						&& !name.startsWith(".") && !name.startsWith("_"));
	}

}