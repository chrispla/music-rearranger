IN:=kikoru.wav
DUR:=60
# CMD:=python rearrange.py --input_audio audio/kikoru.wav --plot --target_time 60
CMD=python rearrange.py --input_audio audio/${IN} --plot --target_time ${DUR} --seg_method=fast

T_CMD=python rearrange.py --input_audio ${IN} --plot --target_time ${DUR}


run: 
	docker run --rm -v .:/app -it --name plac-test plac $(CMD)

testlist: audio/testlist/*.wav
	@echo "Processing $^ ||| $@"
	docker run --rm -v .:/app --name plac-test plac python rearrange.py --input_audio "$^" --plot --target_time 6 --seg_method=fast
	docker run --rm -v .:/app --name plac-test plac python rearrange.py --input_audio $@ --plot --target_time 15 --seg_method=fast
	docker run --rm -v .:/app --name plac-test plac python rearrange.py --input_audio $@ --plot --target_time 30 --seg_method=fast
	docker run --rm -v .:/app --name plac-test plac python rearrange.py --input_audio $@ --plot --target_time 60 --seg_method=fast

build:
	docker build -t plac .

apple:
	docker build --progress=plain -t plac  --platform=linux/aarch64 .
