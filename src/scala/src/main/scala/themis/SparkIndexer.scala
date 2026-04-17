package themis

final case class SparkIndexJob(inputPath: String, outputPath: String, partitions: Int, format: String = "json")

object SparkIndexer {
  def describe(job: SparkIndexJob): String =
    s"Spark indexing plan: input=${job.inputPath}, output=${job.outputPath}, partitions=${job.partitions}, format=${job.format}"

  def planBatches(inputs: Seq[String], outputRoot: String, partitionsPerBatch: Int): Seq[SparkIndexJob] =
    inputs.zipWithIndex.map { case (input, idx) =>
      SparkIndexJob(input, s"$outputRoot/batch_$idx", partitionsPerBatch)
    }

  def akkaDispatchHint(job: SparkIndexJob): String =
    s"Use Akka workers to fan out ${job.partitions} partitions for ${job.inputPath}"
}
