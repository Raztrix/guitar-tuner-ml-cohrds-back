from pydub import AudioSegment

# Load the .webm file (must exist in the same directory)
input_path = "input_audio.webm"
output_path = "output_audio.wav"

try:
    # Read the .webm file
    audio = AudioSegment.from_file(input_path, format="webm")

    # Export it as .wav
    audio.export(output_path, format="wav")

    print(f"✅ Conversion successful! Saved to: {output_path}")

except Exception as e:
    print("❌ Conversion failed.")
    print(e)
