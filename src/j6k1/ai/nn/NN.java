package j6k1.ai.nn;

public class NN {
	private final NNUnit[] units;
	private final double[][][] layers;

	public NN(NNUnit[] units, IInputReader reader, IInitialDataCreator initialDataCreator)
	{
		if(units.length < 3)
		{
			throw new InvalidConfigurationException(
					"Parameter of layer number of multilayer perceptron is incorrect (less than 3)");
		}
		else if(!(units[0].f instanceof FIdentity))
		{
			throw new InvalidConfigurationException(
					"Activation functions other than FIdentity can not be specified for the input layer.");
		}
		this.units = units;

		if(reader.sourceExists())
		{
			double[][][] layers = new double[units.length - 1][][];

			try {
				for(int i=0; i < units.length - 1; i++)
				{
					layers[i] = reader.readVec(units[i].size, units[i+1].size);
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			} finally {
				try {
					reader.close();
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
			}
			this.layers = layers;
		}
		else
		{
			this.layers = initialDataCreator.create();
		}

		if(this.layers.length != units.length - 1)
		{
			throw new InvalidConfigurationException(
					"The layers count do not match. (units = " + units.length + ", layers = " + this.layers.length + ")");
		}
		else
		{
			for(int i=0, il=this.layers.length; i < il; i++)
			{
				if(units[i].size != this.layers[i].length)
				{
					throw new InvalidConfigurationException(
							"The units count do not match. (correct size = " + units[i].size + ", size = " + this.layers[i].length + ")");
				}

				for(int j=0, jl=units[i].size; j < jl; j++)
				{
					if(this.layers[i][j] == null)
					{
						throw new InvalidConfigurationException(
							String.format("Reference to unit %d of layer %d is null.",
								j, i
							));
					}
					else if(this.layers[i][j].length != units[i+1].size)
					{
						throw new InvalidConfigurationException(
							String.format(
								"Weight %d is defined for unit %d in layer %d, but this does not match the number of units in the lower layer.",
								this.layers[i][j].length, i, units[i+1].size
							)
						);
					}
				}
			}
		}
	}

	private NN(NNUnit[] units, double[][][] initialData)
	{
		this.units = units;
		this.layers = initialData;
	}

	private <T> T apply(int[] input, IAfterProcess<T> after)
	{
		if(input.length != units[0].size - 1)
		{
			throw new InvalidStateException(
				"The inputs to the input layer is invalid (the count of inputs must be the count of units -1)");
		}

		double[][] u = new double[units.length][];
		double[][] o = new double[units.length][];

		o[0] = new double[units[0].size];

		o[0][0] = 1.0;

		for(int j=1, J=units[0].size; j < J; j++)
		{
			o[0][j] = (double)input[j-1];
		}

		u[1] = new double[units[1].size];

		for(int k=1, K=units[1].size; k < K; k++)
		{
			u[1][k] += layers[0][0][k];
		}

		for(int j=1, J=units[0].size; j < J; j++)
		{
			for(int k=1, K=units[1].size; k < K; k++)
			{
				u[1][k] += (input[j-1] * layers[0][j][k]);
			}
		}

		o[1] = new double[units[1].size];

		for(int l=1, L=units.length - 1; l < L; l++)
		{
			final int ll = l + 1;

			u[ll] = new double[units[ll].size];
			IActivateFunction f = units[l].f;

			for(int j=0, J = units[l].size; j < J; j++)
			{
				o[l][j] = f.apply(u[l][j]);
			}

			o[ll] = new double[units[ll].size];

			for(int k=1, K = units[ll].size; k < K; k++)
			{
				u[ll][k] = (o[ll][k] += layers[l][0][k]);
			}

			for(int j=1, J=units[l].size; j < J; j++)
			{
				for(int k=1, K = units[ll].size; k < K; k++)
				{
					u[ll][k] = (o[ll][k] += o[l][j] * layers[l][j][k]);
				}
			}
		}

		double[] r = new double[units[units.length-1].size];

		IActivateFunction f = units[units.length-1].f;

		for(int k=1, l=units.length-2, ll=l+1, K = units[ll].size; k < K; k++)
		{
			u[ll][k] = (o[ll][k] += layers[l][0][k]);
		}

		for(int j=1, l=units.length-2, ll=l+1, J=units[l].size; j < J; j++)
		{
			for(int k=1, K = units[ll].size; k < K; k++)
			{
				u[ll][k] = o[ll][k] += (o[l][j] * layers[l][j][k]);
			}
		}

		for(int j=0, l=units.length-1, J=units[units.length-1].size; j < J; j++)
		{
			r[j] = o[l][j] = f.apply(u[l][j]);
		}

		return after.apply(r, o, u);
	}

	public double[] solve(int[] input)
	{
		return apply(input, (r, o, u) -> new DoubleArray(r)).arr;
	}

	public NN learn(int[] input, double[] t)
	{
		if(t.length != units[units.length-1].size)
		{
			throw new InvalidStateException(
				"The number of answers input does not match the number of output units. You must enter data of the same count.");
		}

		return apply(input, (r, o, u) -> {

			double[][][] layers = new double[units.length-1][][];
			double[] d = new double[units[units.length-1].size];

			for(int l=0, L = units.length-1; l < L; l++)
			{
				layers[l] = new double[units[l].size][];
			}

			IActivateFunction f = units[units.length-1].f;

			for(int j=0, l=units.length-2, ll=l+1, J=units[l].size; j < J; j++)
			{
				layers[l][j] = new double[units[ll].size];
			}

			for(int k=0,
					hl=units.length-2,
					l=units.length-1,
					K=units[l].size; k < K; k++)
			{
				d[k] = (r[k] - t[k]) * f.derive(u[l][k]);

				for(int j=0, J=units[hl].size; j < J; j++)
				{
					layers[hl][j][k] = this.layers[hl][j][k] - 0.5 * d[k] * o[hl][j];
				}
			}

			for(int l=units.length - 2; l >= 1; l--)
			{
				final int hl = l - 1;
				final int ll = l + 1;

				f = units[l].f;

				double[] nd = new double[units[l].size];

				for(int i=0, I=units[hl].size; i < I; i++)
				{
					layers[hl][i] = new double[units[l].size];
				}

				for(int j=0, J=units[l].size; j < J; j++)
				{
					for(int k=0, K=units[ll].size; k < K; k++)
					{
						nd[j] += (this.layers[l][j][k] * d[k]);
					}

					nd[j] = nd[j] * f.derive(u[l][j]);

					for(int i=0, I=units[hl].size; i < I; i++)
					{
						layers[hl][i][j] = this.layers[hl][i][j] - 0.5 * nd[j]* o[hl][i];
					}
				}

				d = nd;
			}

			return new NN(units, layers);
		});
	}

	public double[][][] getClonedLayers()
	{
		double[][][] layers = new double[this.layers.length][][];

		for(int l=0, L=this.layers.length; l < L; l++)
		{
			layers[l] = new double[this.layers[l].length][];

			for(int j=0, J=this.layers[l].length; j < J; j++)
			{
				layers[l][j] = new double[this.layers[l][j].length];

				for(int k=0, K=this.layers[l][j].length; k < K; k++)
				{
					layers[l][j][k] = this.layers[l][j][k];
				}
			}
		}

		return layers;
	}
}
class DoubleArray {
	public final double[] arr;

	public DoubleArray(double[] arr)
	{
		this.arr = arr;
	}
}
