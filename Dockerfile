FROM golang:1.26.1-alpine3.23 AS install-dependabot
RUN go install github.com/dependabot/cli/cmd/dependabot@latest \
    && cp $GOPATH/bin/dependabot /usr/local/bin/

FROM docker:29-dind AS runtime
COPY --from=install-dependabot /usr/local/bin/dependabot /usr/local/bin/
ENTRYPOINT ["dependabot"]