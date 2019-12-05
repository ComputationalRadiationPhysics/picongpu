#/usr/bin/env bash

show_directory()
{
    # Autocompletion from the bash "cd" command
    # Copied from "bash-completion(1:2.8-1ubuntu1)" in /usr/share/bash-completion/bash_completion
    local cur prev words cword
    _init_completion || return

    local IFS=$'\n' i j k

    compopt -o filenames

    # Show example directories with autocompletion for pic-create
    if [[ $1 == "examples" ]]; then
        for i in ${PIC_EXAMPLES//:/$'\n'}; do
            k="${#COMPREPLY[@]}"
            for j in $( compgen -d -- $i/$cur ); do
                if [[ ( ! -d ${j#$i/} ) ]]; then
                  j+="/"
                fi
                COMPREPLY[k++]=${j#$i/}
            done
        done
    fi

    # Use standard dir completion if no CDPATH or parameter starts with /,
    # ./ or ../
    if [[ -z "${CDPATH:-}" || "$cur" == ?(.)?(.)/* ]]; then
        _filedir -d
        return
    fi

    local -r mark_dirs=$(_rl_enabled mark-directories && echo y)
    local -r mark_symdirs=$(_rl_enabled mark-symlinked-directories && echo y)

    # we have a CDPATH, so loop on its contents
    for i in ${CDPATH//:/$'\n'}; do
        # create an array of matched subdirs
        k="${#COMPREPLY[@]}"
        for j in $( compgen -d -- $i/$cur ); do
            if [[ ( $mark_symdirs && -h $j || $mark_dirs && ! -h $j ) && ! -d ${j#$i/} ]]; then
                j+="/"
            fi
            COMPREPLY[k++]=${j#$i/}
        done
    done

    _filedir -d

    # if only one possible result then autocomplete to this result and enter the directory
    if [[ ${#COMPREPLY[@]} -eq 1 ]]; then
        i=${COMPREPLY[0]}
        if [[ "$i" == "$cur" && $i != "*/" ]]; then
            COMPREPLY[0]="${i}/"
        fi
    fi

    return 0
}

show_parameters()
{
    local opts=$1
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}))
    return 0
}

_pic-build()
{
    # Load current string
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    if [[ "${prev}" == "-b" || "${prev}" == "--backend" ]]; then
        # available backends from documentation
        show_parameters "cuda omp2b serial tbb threads"
        return 0
    fi

    if [[ "${prev}" == "-c" || "${prev}" == "--cmake" ]]; then
        show_parameters "-b --backend -c --cmake -t -f --force -h --help"
        return 0 # @TODO show available cmake variables
    fi

    if [[ "${prev}" == "-t" ]]; then
        show_parameters "-b --backend -c --cmake -t -f --force -h --help"
        return 0 # @TODO show available preset from cmakeFlags
    fi

    show_parameters "-b --backend -c --cmake -t -f --force -h --help"
    return 0
}

_pic-create()
{
    # load current string
    local cur
    cur="${COMP_WORDS[COMP_CWORD]}"
    case "$cur" in
        # check if it is a parameter
        -*)
            show_parameters "-f --force -h --help"
            return 0
            ;;
        # default case list directory content
        *)
            show_directory "examples"
            return 0
            ;;
    esac
}

_pic-compile()
{
    # load current string
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    if [[ "${prev}" == "-c" || "${prev}" == "--cmake" ]]; then
        return 0 # @TODO show available cmake variables
    fi

    case "$cur" in
        # check if it is a parameter
        -*)
            show_parameters "-l -q -j -c -ccmake -h --help"
            return 0
            ;;
        *)
            show_directory
            return 0
            ;;
    esac
}

_pic-configure()
{
    # Load current string
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    if [[ "${prev}" == "-i" || "${prev}" == "--install" ]]; then
        show_directory
    fi

    if [[ "${prev}" == "-b" || "${prev}" == "--backend" ]]; then
        show_directory
        return 0 # @TODO show available backends
    fi

    if [[ "${prev}" == "-c" || "${prev}" == "--cmake" ]]; then
        show_directory
        return 0 # @TODO show available cmake variables
    fi

    if [[ "${prev}" == "-t" ]]; then
        show_directory
        return 0 # @TODO show available preset from cmakeFlags
    fi

    case "$cur" in
        # check if it is a parameter
        -*)
            show_parameters "-b --backend -c --cmake -t -f --force -h --help"
            return 0
            ;;
        *)
            show_directory
            return 0
            ;;
    esac
}

_pic-edit()
{
    local cur names
    cur="${COMP_WORDS[COMP_CWORD]}"

    # Find all param files
    names=$(for x in $(ls -1 $PICSRC/include/picongpu/param/*.param); do echo ${x} ; done )
    # Remove .param from param file
    names=$(for x in $names; do echo ${x%".param"}; done)
    # Remove filepath from param file
    names=$(for x in $names; do echo ${x#"$PICSRC/include/picongpu/param/"}; done)
    COMPREPLY=( $(compgen -W "${names}" -- ${cur}) )
    return 0
}

_tbg()
{
    # Load current string
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # List available config files in default directory (inspired by https://debian-administration.org/article/317/An_introduction_to_bash_completion_part_2)
    if [[ "${prev}" == "-c" || "${prev}" == "--config" ]]; then
        local names=$(for x in $(ls -1 etc/picongpu/*.cfg); do echo ${x/\/etc\/picongpu\//} ; done )
        COMPREPLY=( $(compgen -W "${names}" -- ${cur}) )
        return 0
    fi

    if [[ "${prev}" == "-s" || "${prev}" == "--submit" ]]; then
        case "$cur" in
            # check if it is a parameter
            -*)
                show_parameters "-c --config -t --tpl -o -f --force"
                return 0
                ;;
            *)
                # show submit systems which are listed in the docs
                show_parameters "sbatch qsub bsub"
                return 0
                ;;
        esac
        return 0
    fi

    if [[ "${prev}" == "-t" || "${prev}" == "--tpl" ]]; then
        show_directory
        return 0 # @TODO show template file
    fi

    if [[ "${prev}" == "-o" ]]; then
        show_directory
        return 0 # @TODO show template variables
    fi

    case "$cur" in
        # check if it is a parameter
        -*)
            show_parameters "-c --config -s --submit -t --tpl -o -f --force -h --help"
            return 0
            ;;
        *)
            show_directory
            return 0
            ;;
    esac
}

# Invoke the responding function calls for completion of pic-commands
complete -F _pic-build pic-build
complete -o nospace -F _pic-create pic-create
complete -o nospace -F _pic-compile pic-compile
complete -o nospace -F _pic-configure pic-configure
complete -F _pic-edit pic-edit
complete -o nospace -F _tbg tbg

