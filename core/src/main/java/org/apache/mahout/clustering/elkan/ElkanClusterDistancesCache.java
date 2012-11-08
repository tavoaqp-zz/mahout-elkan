package org.apache.mahout.clustering.elkan;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.spectral.common.VectorCache;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

/**
 * Follows the same pattern as VectorCache but it keeps many vectors cached in a
 * LRU data structure bounded by a fixed size.
 * 
 * @author gustavo
 * 
 */
public class ElkanClusterDistancesCache {

	public static String CACHED_VECTOR_SUFFIX = "vector-";
	public static String CACHED_OLD_DISTANCE_SUFFIX = "oldDistance-";
	public static String CACHED_HALF_DISTANCE_SUFFIX = "halfDistance-";

	private ArrayList<URI> vectorURIList;
	private LRUVectorCache lruCache;
	private Path cachePath;
	private Configuration conf;
	private boolean overwritePath;
	private boolean deleteOnExit;
	private Vector oldClusterDistances;
	private Vector halfClusterDistances;

	private static final Logger log = LoggerFactory
			.getLogger(VectorCache.class);

	public ElkanClusterDistancesCache(int numClusters, Configuration conf,
			Path cachePath) {
		this(numClusters, conf, cachePath, false, false);
	}

	public ElkanClusterDistancesCache(int numClusters, Configuration conf,
			Path cachePath, boolean overwritePath, boolean deleteOnExit) {
		vectorURIList = new ArrayList<URI>();
		lruCache = new LRUVectorCache(numClusters);
		this.conf = conf;
		this.cachePath = cachePath;
		this.overwritePath = overwritePath;
		this.deleteOnExit = deleteOnExit;
	}

	public void save(IntWritable key, Vector vector) throws IOException {		
		save(CACHED_VECTOR_SUFFIX, key, vector);
	}
	
	public void save(String suffix, IntWritable key, Vector vector) throws IOException {
		String fileName=suffix;
		if (key!=null)
			fileName+=key.get();
		else
			key=new IntWritable(0);
		Path output = new Path(cachePath, fileName);
		FileSystem fs = FileSystem.get(output.toUri(), conf);
		output = fs.makeQualified(output);
		if (overwritePath) {
			HadoopUtil.delete(conf, output);
		}

		vectorURIList.add(output.toUri());

		SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, output,
				IntWritable.class, VectorWritable.class);
		try {
			writer.append(key, new VectorWritable(vector));
		} finally {
			Closeables.closeQuietly(writer);
		}

		if (deleteOnExit) {
			fs.deleteOnExit(output);
		}
	}

	public Vector get(Integer index) {
		if (lruCache.containsKey(index))
			return lruCache.get(index);
		else {
			Vector vector;
			try {
				vector = loadVector(index);
				lruCache.put(index, vector);
				return vector;
			} catch (IOException e) {
				log.error("Error loading cached vector with ID:" + index);
				return null;
			}

		}
	}

	private Vector loadVector(Integer index) throws IOException {
		return loadVector(CACHED_VECTOR_SUFFIX , index);
	}
	
	private Vector loadVector(String prefix, Integer index) throws IOException {
		String fileName=prefix;
		if (index!=null)
			fileName+=index.toString();
		Path vectorPath = new Path(cachePath, fileName);
		SequenceFileValueIterator<VectorWritable> iterator = new SequenceFileValueIterator<VectorWritable>(
				vectorPath, true, conf);
		try {
			return iterator.next().get();
		} finally {
			Closeables.closeQuietly(iterator);
		}
	}

	public void preload() {
		for (int vectorIndex = 0; vectorIndex < Math
				.max(1, lruCache.size() / 2); vectorIndex++) {
			get(vectorIndex);
		}
	}

	public void cleanup() {
		lruCache.clear();
	}

	public void saveOldClusterDistances(Vector oldClusterDistances)
			throws IOException {
		save(CACHED_OLD_DISTANCE_SUFFIX, null, oldClusterDistances);
	}

	public Vector getOldClusterDistances() {
		if (oldClusterDistances == null) {
			try {
				oldClusterDistances=loadVector(CACHED_OLD_DISTANCE_SUFFIX,null);
			} catch (IOException e1) {
				log.error("Error getting Old Cluster Distances vector",e1);
			}		
		}
		return oldClusterDistances;
	}

	public void saveHalfDistancesVector(Vector halfClusterDistances)
			throws IOException {		
		save(CACHED_HALF_DISTANCE_SUFFIX, null, halfClusterDistances);
	}
	
	public Vector getHalfClusterDistances() {
		if (halfClusterDistances == null) {			
			try {
				halfClusterDistances=loadVector(CACHED_HALF_DISTANCE_SUFFIX,null);
			} catch (IOException e1) {
				log.error("Error getting Half Cluster Distances vector",e1);
			}			
		}
		return halfClusterDistances;
	}
}

class LRUVectorCache extends LinkedHashMap<Integer, Vector> {
	private int maxEntries;

	public LRUVectorCache() {
		this(0);
	}

	public LRUVectorCache(int maxEntries) {
		super(maxEntries + 1, 1.0f, true);
		this.maxEntries = maxEntries;
	}

	@Override
	protected boolean removeEldestEntry(Entry<Integer, Vector> eldest) {
		return super.size() > maxEntries;
	}

}