package j6k1.ai.nn;

public interface IInputReader {
	public boolean sourceExists();
	public double[][] readVec(final int a, final int b);
	public boolean close();
}
