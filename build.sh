version=$(cat version)
echo $version

docker build -t akrasnov87/ai-video-transcriber:$version .