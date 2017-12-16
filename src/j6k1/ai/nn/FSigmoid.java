package j6k1.ai.nn;

public final class FSigmoid implements IActivateFunction {
	@Override
	public double apply(double u) {
		return 1.0 / (1.0 + Math.exp(-u));
	}

	@Override
	public double derive(double e) {
		e = apply(e);
		return e * (1 - e);
	}
}
