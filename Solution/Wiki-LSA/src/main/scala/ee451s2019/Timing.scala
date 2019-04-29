package ee451s2019

class Timing {
	private val MESSAGE = "[%s] takes %f seconds."

	private var startTime = System.currentTimeMillis()

	def restart() = {
		startTime = System.currentTimeMillis()
	}

	def stop(text: String = "") = {
		val seconds = (System.currentTimeMillis() - startTime) / 1000.0
		println(MESSAGE.format(text, seconds))
	}
}
