from __future__ import annotations

from llm_pipeline_bench.profile_runner import main


class _Utils:
    def __init__(self, profiler):
        self.profiler = profiler

    def run(self, statement, filename=None, sort=-1):
        profiler = self.profiler()
        try:
            profiler.run(statement)
        except SystemExit:
            pass
        if filename is not None:
            profiler.dump_stats(filename)
        else:
            profiler.print_stats(sort)

    def runctx(self, statement, globals_dict, locals_dict, filename=None, sort=-1):
        profiler = self.profiler()
        try:
            profiler.runctx(statement, globals_dict, locals_dict)
        except SystemExit:
            pass
        if filename is not None:
            profiler.dump_stats(filename)
        else:
            profiler.print_stats(sort)


def run(statement, filename=None, sort=-1):
    import cProfile

    return _Utils(cProfile.Profile).run(statement, filename, sort)


def runctx(statement, globals_dict, locals_dict, filename=None, sort=-1):
    import cProfile

    return _Utils(cProfile.Profile).runctx(statement, globals_dict, locals_dict, filename, sort)


if __name__ == "__main__":
    main()
