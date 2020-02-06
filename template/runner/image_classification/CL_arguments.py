# DeepDIVA
from template.runner.base import BaseCLArguments

class CLArguments(BaseCLArguments):

    def parse_arguments(self, args=None):
        args, self.parser = super().parse_arguments(args)

        # Set default value for --split-type in _darwin_options()
        if args.split_type is None:
            args.split_type = "stratified_tag"

        return args, self.parser