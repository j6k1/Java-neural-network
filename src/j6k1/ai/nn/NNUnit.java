package j6k1.ai.nn;

public class NNUnit {
	public final int size;
	public final IActivateFunction f;

	public NNUnit(final int size)
	{
		this(size, new FIdentity());
	}

	public NNUnit(final int size, IActivateFunction f)
	{
		this.size = size;
		this.f = f;
	}
}
