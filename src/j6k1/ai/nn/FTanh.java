package j6k1.ai.nn;

public final class FTanh implements IActivateFunction {
	@Override
	public double apply(double u) {
		return Math.tanh(u);
	}

	@Override
	public double derive(double e) {
		e = Math.tanh(e);
		return (1.0 - e * e);
	}
}
