# Define Docker image names and tags
set(DOCKER_BASE_IMAGE "pveleskopglc/chipstar:base")
set(DOCKER_CPP_LINTER_IMAGE "pveleskopglc/chipstar:cpp-linter")
set(DOCKER_CPP_LATEST_IMAGE "pveleskopglc/chipstar:latest")
set(DOCKER_CPP_FULL_IMAGE "pveleskopglc/chipstar:cpp-full")

# Function to add custom target with visible output
function(add_docker_target target_name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "WORKING_DIRECTORY;COMMENT" "COMMAND")
    add_custom_target(${target_name}
        COMMAND ${CMAKE_COMMAND} -E echo "Executing: ${ARG_COMMAND}"
        COMMAND ${ARG_COMMAND}
        WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY}
        COMMENT ${ARG_COMMENT}
        USES_TERMINAL
        VERBATIM
    )
endfunction()

# Docker build targets
add_docker_target(docker-build-base
    COMMAND docker build -t ${DOCKER_BASE_IMAGE} --progress=plain -f ${CMAKE_SOURCE_DIR}/docker/DockerfileBase ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building base Docker image"
)

add_docker_target(docker-build-cpp-linter
    COMMAND docker build -t ${DOCKER_CPP_LINTER_IMAGE} --progress=plain -f ${CMAKE_SOURCE_DIR}/docker/DockerfileCPPLinter ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building C++ linter Docker image"
    DEPENDS docker-build-base
)

add_docker_target(docker-build-full
    COMMAND docker build -t ${DOCKER_CPP_FULL_IMAGE} --progress=plain -f ${CMAKE_SOURCE_DIR}/docker/DockerfileFull ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building full Docker image"
    DEPENDS docker-build-base
)

add_docker_target(docker-build-latest
    COMMAND docker build -t ${DOCKER_CPP_LATEST_IMAGE} --progress=plain -f ${CMAKE_SOURCE_DIR}/docker/DockerfileLatest ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building latest Docker image"
    DEPENDS docker-build-base
)

add_docker_target(docker-publish-full
    COMMAND docker push ${DOCKER_CPP_FULL_IMAGE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Publishing full Docker image"
    DEPENDS docker-build-full
)

add_docker_target(docker-publish-latest
    COMMAND docker push ${DOCKER_CPP_LATEST_IMAGE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Publishing latest Docker image"
    DEPENDS docker-build-latest
)

# Docker publish targets
add_docker_target(docker-publish-base
    COMMAND docker push ${DOCKER_BASE_IMAGE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Publishing base Docker image"
    DEPENDS docker-build-base
)

add_docker_target(docker-publish-cpp-linter
    COMMAND docker push ${DOCKER_CPP_LINTER_IMAGE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Publishing C++ linter Docker image"
    DEPENDS docker-build-cpp-linter
)

# Convenience targets
add_docker_target(docker-build-all
    COMMAND ${CMAKE_COMMAND} -E echo "Building all Docker images"
    DEPENDS docker-build-base docker-build-cpp-linter docker-build-full docker-build-latest
    COMMENT "Building all Docker images"
)

add_docker_target(docker-publish-all
    COMMAND ${CMAKE_COMMAND} -E echo "Publishing all Docker images"
    DEPENDS docker-publish-base docker-publish-cpp-linter docker-publish-full docker-publish-latest
    COMMENT "Publishing all Docker images"
)

add_docker_target(docker-all
    COMMAND ${CMAKE_COMMAND} -E echo "Building and publishing all Docker images"
    DEPENDS docker-build-all docker-publish-all
    COMMENT "Building and publishing all Docker images"
)
