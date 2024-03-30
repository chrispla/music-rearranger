import subprocess
from pathlib import Path

CMD = (
    "docker run --rm --gpus all -v .:/app --name plac-test-{id} plac "
    "python rearrange.py "
    "--input_audio \"{audio_file}\" "
    "--plot --target_time {duration} "
    "--seg_method=fast"
)


def render_cutdowns(audio_file: str):
    print(f"Processing {audio_file}...")
    for dur in [6, 15, 30, 60]:
        cmd = CMD.format(id=dur, audio_file=audio_file, duration=dur)
        print(cmd)
        subprocess.run(cmd ,shell=True)

def get_audio_files():
    p = Path(".") / "audio" / "testlist"
    files = list(p.glob("*.wav"))
    print(f"Found {len(files)} number of wav files")
    return files

def render_testlist():
    audio_files = get_audio_files()
    for af in audio_files:
        render_cutdowns(af)

if __name__ == "__main__":
    # print(CMD.format(audio_file="ad", duration="2"))
    render_testlist()
