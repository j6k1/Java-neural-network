package j6k1.ai.nn;

import java.util.Arrays;

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
			for(int i=0, l=this.layers.length; i < l; i++)
			{
				if(units[i].size != this.layers[i].length)
				{
					throw new InvalidConfigurationException(
							"The units count do not match. (correct size = " + units[i].size + ", size = " + this.layers[i].length + ")");
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
		double[][] weighted = new double[units.length-1][];
		double[][] output = new double[units.length-1][];

		for(int j=0, jl=units[1].size; j < jl; j++)
		{
			weighted[0][j] += layers[0][0][j];
		}

		for(int i=1, il=input.length; i < il; i++)
		{
			for(int j=0, jl=units[1].size; j < jl; j++)
			{
				weighted[0][j] += input[i] * layers[0][i][j];
			}
		}

		for(int i=1, il=units.length - 1; i < il; i++)
		{
			output[i-1] = new double[units[i].size];

			IActivateFunction f = units[i].f;

			for(int k=1, kl = units[i].size; k < kl; k++)
			{
				output[i-1][k] = f.apply(weighted[i-1][k]);
			}

			for(int j=0, jl=units[i].size; j < jl; j++)
			{
				output[i-1][0] = layers[i][j][0];

				for(int k=1, kl = units[i+1].size; k < kl; k++)
				{
					output[i][k] += output[i-1][j] * layers[i][j][k];
				}
			}
		}

		double[] result = new double[units[units.length-1].size];
		IActivateFunction f = units[units.length-1].f;

		for(int i=0, l=result.length, ll=output.length-1; i < l; i++)
		{
			result[i] = f.apply(output[ll][i]);
		}

		return after.apply(result, output, weighted);
	}

	public double[] solve(int[] input)
	{
		return apply(input, (result, output, weighted) -> new DoubleArray(result)).arr;
	}

	public NN learn(int[] input, double[] t)
	{
		return apply(input, (result, output, weighted) -> {

			double[][][] layers = new double[units.length-1][][];
			double[] delta = new double[units[units.length-1].size];

			layers[units.length-2] = new double[units[units.length-1].size][];

			IActivateFunction f = units[units.length-1].f;

			for(int j=0,
					wl=weighted.length,
					cl=layers.length-1,
					ol=output.length-1,
					jl=units[units.length-1].size; j < jl; j++)
			{
				delta[j] = (result[j] - t[j]) * f.derive(weighted[wl-1][j]);

				for(int k=0, kl=units[units.length-2].size; k < kl; k++)
				{
					layers[cl][k][j] = this.layers[cl][k][j] - 0.5 * delta[j] * output[ol][k];
				}
			}

			for(int i=units.length - 1; i >= 1; i--)
			{
				double[] nextdelta = new double[i-1];

				for(int j=0, jl=units[i].size; j < jl; j++)
				{
					for(int k=0, kl=units[i-1].size; k < kl; k++)
					{
						nextdelta[j] += this.layers[i-1][j][k] * delta[j];
					}

					for(int k=0, kl=units[i-1].size; k < kl; k++)
					{
						nextdelta[j] = nextdelta[j] * weighted[i-1][k];
						layers[i-1][k][j] = this.layers[i-1][k][j] - 0.5 * nextdelta[j] * output[k][j];
					}
				}
			}

			return new NN(units, layers);
		});
	}

	public double[][][] getClonedLayers()
	{
		double[][][] layers = new double[this.layers.length][][];

		for(int i=0, il=this.layers.length; i < il; i++)
		{
			layers[i] = new double[this.layers[i].length][];

			for(int j=0, jl=this.layers[i].length; j < jl; j++)
			{
				layers[i][j] = new double[this.layers[i][j].length];

				for(int k=0, kl=this.layers.length; k < kl; k++)
				{
					layers[i][j][k] = this.layers[i][j][k];
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
