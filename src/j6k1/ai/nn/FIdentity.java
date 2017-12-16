package j6k1.ai.nn;

public final class FIdentity implements IActivateFunction {
	@Override
	public double apply(double u) {
		return u;
	}

	@Override
	public double derive(double e) {
		return 1;
	}
}
