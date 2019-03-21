## Run linter
### use circleci

Setup

```
$ curl -o /usr/local/bin/circleci https://circle-downloads.s3.amazonaws.com/releases/build_agent_wrapper/circleci
$ chmod +x /usr/local/bin/circleci
$ circleci update
$ circleci switch
```

Run

```
$ circleci build  //Please make sure docker is installed. 
```

### not use circleci
Setup

```
pip3 install flake8
pip3 install autopep8
```

Run

```
flake8 ./*.py   // check lint
autopep8 -r . -i   // fix lint automatically
```