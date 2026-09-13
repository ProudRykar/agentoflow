from dataclasses import MISSING, fields, is_dataclass

from core.entities.models.builtin.read_file import ReadFileInput
from core.entities.models.tool import InputT
from typing import get_type_hints


class ArgumentDecoderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        field: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.field = field
        self.expected = expected
        self.actual = actual


class ArgumentDecoderErrorFactory:
    @staticmethod
    def invalid_type(
        field: str,
        expected: str,
        actual: str,
    ) -> ArgumentDecoderError:
        """Ошибка всплывающая при неверном типе поля.
        
        Args:
            field (str): Поле класса.
            expected (str): Ожидаемый тип поля.
            actual (str): Переданный неверный тип поля.
        Returns:
            ArgumentDecoderError: Отчёт о ошибке
        """

        return ArgumentDecoderError(
            f"Invalid type for field '{field}': "
            f"expected {expected}, got {actual}",
            code="invalid_type",
            field=field,
            expected=expected,
            actual=actual,
        )
    

    @staticmethod
    def missing_field(
        field: str,
    ) -> ArgumentDecoderError:
        """Ошибка всплывающая при отсутсвующем поле."""

        return ArgumentDecoderError (
            f"Missing field '{field}'",
            code="missing_field",
            field=field
        )
    

    @staticmethod
    def unknown_field(
        field: str,
    ) -> ArgumentDecoderError:
        """Ошибка всплывающая при передаче несуществующего поля."""

        return ArgumentDecoderError(
            f"Unknown field '{field}'",
            code="unknown_field",
            field=field,
        )


class ArgumentDecoder:
    def decode(
        self,
        data: dict[str, object],
        input_type: type[InputT],
    ) -> InputT:

        # Проврека на то, является ли input_type датаклассом.
        if not is_dataclass(input_type):
            raise ArgumentDecoderError(
                f"{input_type.__name__} must be a dataclass",
                code="invalid_input_type",
            )

        input_fields = fields(input_type)
        field_names = {field.name for field in input_fields}

        # Проверка на то, есть ли неверные поля.
        for name in data:
            if name not in field_names:
                raise ArgumentDecoderErrorFactory.unknown_field(name)

        hints = get_type_hints(input_type)

        for field in input_fields:
            # Проверка на отсутствующие поля.
            if field.name not in data:
                if (
                    field.default is MISSING
                    and field.default_factory is MISSING
                ):
                    raise ArgumentDecoderErrorFactory.missing_field(
                        field=field.name
                    )

                continue
            
            field_type = hints[field.name]
            
            # Проверка на совпадение типов поля.
            if not isinstance(data[field.name], field_type):
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field=field.name,
                    expected=field_type.__name__,
                    actual=type(data[field.name]).__name__,
                )

        return input_type(**data)
