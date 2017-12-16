package j6k1.ai.nn;

@FunctionalInterface
public interface IAfterProcess<T> {
	public T apply(double[] result, double[][] output, double[][] weighted);
}
