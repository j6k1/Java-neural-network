package j6k1.ai.nn;

public final class FReLU implements IActivateFunction {
	@Override
	public double apply(double u) {
		return u > 0 ? u : 0;
	}

	@Override
	public double derive(double e) {
		return e > 0 ? 1 : 0;
	}
}
