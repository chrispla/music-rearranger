IN:=kikoru.wav
DUR:=30
# CMD:=python rearrange.py --input_audio audio/kikoru.wav --plot --target_time 60
CMD=python rearrange.py --input_audio audio/${IN} --plot --target_time ${DUR}


run: 
	docker run --rm -v .:/app --name plac-test plac $(CMD)

build:
	docker build -t plac .
