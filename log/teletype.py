"""
-------------------------------------------------------------------------------------
Exploring Linguistically Enriched Transformers for Low-Resource Relation Extraction:
    --Enriched Attention on PLM
    
    by Pedro G. Bascoy  (Bosch Center for Artificial Intelligence (BCAI)),
    
    with the supervision of
    
    Prof. Dr. Sebastian Padó (Institut für Machinelle Sprachverarbeitung (IMS)),
    and Dr. Heike Adel-Vu  (BCAI).
-------------------------------------------------------------------------------------
"""
from typing import Dict, Any, Optional, List, Union

"""
teletype class: handles the standard output
"""

class teletype(object):

    # dictionary to map modules with the indentation value
    messages: Dict[Any, List[Union[int, bool]]] = {}
    # current indentation
    idx: int = 0
    # indentation width
    tab_spacing: int = 4

    @staticmethod
    def start_task (message: str,
              class_name: str ) -> None:
        """
        Logs the beginning of a task
        :param message: Message describing the task
        :param class_name: Class responsible of carrying out the task
        :return:
        """

        curr_idx: int

        # retrieve indentation index: add new
        if class_name not in teletype.messages:

            # invalidate
            teletype.invalid()

            curr_idx = teletype.idx

            teletype.messages[class_name] = [teletype.idx, False]
            teletype.idx += 1

        # retrieve indentation index
        else:
            curr_idx, _ = teletype.messages[class_name]

        # print content
        print()
        print(f'{" "*curr_idx*teletype.tab_spacing}@ {class_name}: {message} ... ', end='')


    @staticmethod
    def invalid() -> None:
        """
        Sets the second value to the list of all elements in `messages` to True. Those classes who are assigned a value
        of True, cannot display the OK at the same line.
        :return:
        """
        for key, value in teletype.messages.items():
            value[1] = True

    @staticmethod
    def start_subtask (message: str,
                  class_name: str,
                  task: str) -> None:
        """
        Similar to `start_task`
        :param message: Message describing the subtask
        :param class_name: Class responsible of carrying out the task
        :param task: function responsible of carrying out the task
        :return:
        """

        curr_idx: int

        # create a new id pair
        id: str = f'{class_name}${task}'

        # retrieve indentation index: add new
        if id not in teletype.messages:

            teletype.invalid()

            curr_idx = teletype.idx

            teletype.messages[id] = [teletype.idx, False]
            teletype.idx += 1
        # retrieve indentation index
        else:
            curr_idx, _ = teletype.messages[id]

        # print message
        print()
        print(f'{" "*curr_idx*teletype.tab_spacing}$ {task}: {message} ... ', end='')

    @staticmethod
    def print_information(message: str,
                   class_name: Optional[str] = None) -> None:
        """
        Prints a message in the current task (if class_name is None), or in the class_name indentation space.
        :param message: Message to be printed
        :param class_name: If any, class responsible of the message.
        :return:
        """

        teletype.invalid()

        # get indentation
        if class_name is not None:
            curr_idx, _ = teletype.messages[class_name]
        else:
            curr_idx = teletype.idx - 1

        # print message
        print(f'\n{" " * (curr_idx+1)*teletype.tab_spacing}* {message}', end='')

    @staticmethod
    def finish_subtask(class_name: str,
                    task: str,
                    success: bool = True,
                    message: Optional[str] = None):
        """
        Finishes a subtask
        :param class_name: Class responsible of carrying out the task
        :param task: function responsible of carrying out the task
        :param success: flag indicating whether the subtasks was successfully finished
        :param message: Optional message
        :return:
        """

        teletype.finish_task(f'{class_name}${task}',success, message)

    @staticmethod
    def finish_task(class_name: str,
                    success: bool = True,
                    message: Optional[str] = None):
        """
        Finishes a task
        :param class_name: Class responsible of carrying out the task
        :param success: flag indicating whether the subtasks was successfully finished
        :param message: Optional message
        :return:
        """

        # retrieve index and printing flag
        curr_idx, flag = teletype.messages[class_name]
        nl = '\n'

        # print content
        if success:

            if flag:
                print(f'{nl + " " * curr_idx*teletype.tab_spacing}- {class_name} ... OK', end='')
            else:
                print('OK', end='')

        else:

            if flag:
                print(f'{nl + " " * curr_idx*teletype.tab_spacing} {class_name} ... FAILED', end='')
            else:
                print('FAILED', end='')

        # print optional message
        if message is not None:
            print(f': {message}',end='')

        # update indices
        teletype.idx -= 1
        del teletype.messages[class_name]