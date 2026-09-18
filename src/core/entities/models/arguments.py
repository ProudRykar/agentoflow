from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

from core.entities.models.tool import InputT


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
        return ArgumentDecoderError(
            f"Missing field '{field}'",
            code="missing_field",
            field=field,
        )

    @staticmethod
    def unknown_field(
        field: str,
    ) -> ArgumentDecoderError:
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
        if not is_dataclass(input_type):
            raise ArgumentDecoderError(
                f"{input_type.__name__} must be a dataclass",
                code="invalid_input_type",
            )

        input_fields = fields(input_type)
        field_names = {
            field.name
            for field in input_fields
        }

        for name in data:
            if name not in field_names:
                raise ArgumentDecoderErrorFactory.unknown_field(
                    name,
                )

        hints = get_type_hints(input_type)

        decoded: dict[str, object] = {}

        for field in input_fields:
            if field.name not in data:
                if (
                    field.default is MISSING
                    and field.default_factory is MISSING
                ):
                    raise ArgumentDecoderErrorFactory.missing_field(
                        field=field.name,
                    )

                continue

            field_type = hints[field.name]
            value = data[field.name]

            try:
                value = self._decode_value(
                    value,
                    field_type,
                )
            except (TypeError, ValueError) as exc:
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field=field.name,
                    expected=self._type_name(field_type),
                    actual=type(value).__name__,
                ) from exc

            if not self._matches_type(
                value,
                field_type,
            ):
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field=field.name,
                    expected=self._type_name(field_type),
                    actual=type(value).__name__,
                )

            decoded[field.name] = value

        return input_type(**decoded)

    @classmethod
    def _decode_value(
        cls,
        value: object,
        expected_type: Any,
    ) -> object:
        if expected_type is Any:
            return value

        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Enum:
        #
        # "reasoning" -> ModelCapability.REASONING
        # "complex"   -> ModelComplexity.COMPLEX
        if isinstance(expected_type, type) and issubclass(
            expected_type,
            Enum,
        ):
            if isinstance(value, expected_type):
                return value

            if isinstance(value, str):
                try:
                    return expected_type(value)
                except ValueError:
                    # Иногда LLM может прислать имя enum:
                    # "REASONING" вместо "reasoning".
                    try:
                        return expected_type[value]
                    except KeyError:
                        pass

            return value

        # Optional[T] / Union[T, U]
        if origin in (UnionType,):
            for arg in args:
                if arg is type(None) and value is None:
                    return None

                try:
                    decoded = cls._decode_value(
                        value,
                        arg,
                    )
                except (TypeError, ValueError):
                    continue

                if cls._matches_type(
                    decoded,
                    arg,
                ):
                    return decoded

            return value

        if str(origin) == "typing.Union":
            for arg in args:
                if arg is type(None) and value is None:
                    return None

                try:
                    decoded = cls._decode_value(
                        value,
                        arg,
                    )
                except (TypeError, ValueError):
                    continue

                if cls._matches_type(
                    decoded,
                    arg,
                ):
                    return decoded

            return value

        # list[T]
        if origin is list:
            if not isinstance(value, list):
                return value

            if not args:
                return value

            return [
                cls._decode_value(
                    item,
                    args[0],
                )
                for item in value
            ]

        # set[T]
        if origin is set:
            if not isinstance(value, (set, list, tuple)):
                return value

            if not args:
                return value

            return {
                cls._decode_value(
                    item,
                    args[0],
                )
                for item in value
            }

        # tuple[T, ...]
        if origin is tuple:
            if not isinstance(value, (tuple, list)):
                return value

            if not args:
                return value

            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(
                    cls._decode_value(
                        item,
                        args[0],
                    )
                    for item in value
                )

            if len(value) != len(args):
                return value

            return tuple(
                cls._decode_value(
                    item,
                    item_type,
                )
                for item, item_type in zip(
                    value,
                    args,
                    strict=True,
                )
            )

        # dict[K, V]
        if origin is dict:
            if not isinstance(value, dict):
                return value

            if len(args) != 2:
                return value

            key_type, value_type = args

            return {
                cls._decode_value(
                    key,
                    key_type,
                ): cls._decode_value(
                    item,
                    value_type,
                )
                for key, item in value.items()
            }

        return value

    @classmethod
    def _matches_type(
        cls,
        value: object,
        expected_type: Any,
    ) -> bool:
        if expected_type is Any:
            return True

        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Enum
        if isinstance(expected_type, type) and issubclass(
            expected_type,
            Enum,
        ):
            return isinstance(
                value,
                expected_type,
            )

        # Обычный runtime-класс.
        if origin is None:
            if expected_type is None:
                return value is None

            if not isinstance(expected_type, type):
                return True

            return isinstance(
                value,
                expected_type,
            )

        # Union / Optional.
        if origin in (UnionType,):
            return any(
                cls._matches_type(
                    value,
                    arg,
                )
                for arg in args
            )

        if str(origin) == "typing.Union":
            return any(
                cls._matches_type(
                    value,
                    arg,
                )
                for arg in args
            )

        # list[T]
        if origin is list:
            if not isinstance(value, list):
                return False

            if not args:
                return True

            return all(
                cls._matches_type(
                    item,
                    args[0],
                )
                for item in value
            )

        # set[T]
        if origin is set:
            if not isinstance(value, set):
                return False

            if not args:
                return True

            return all(
                cls._matches_type(
                    item,
                    args[0],
                )
                for item in value
            )

        # tuple[T, ...] / tuple[T, U]
        if origin is tuple:
            if not isinstance(value, tuple):
                return False

            if not args:
                return True

            if len(args) == 2 and args[1] is Ellipsis:
                return all(
                    cls._matches_type(
                        item,
                        args[0],
                    )
                    for item in value
                )

            if len(value) != len(args):
                return False

            return all(
                cls._matches_type(
                    item,
                    item_type,
                )
                for item, item_type in zip(
                    value,
                    args,
                    strict=True,
                )
            )

        # dict[K, V]
        if origin is dict:
            if not isinstance(value, dict):
                return False

            if len(args) != 2:
                return True

            key_type, value_type = args

            return all(
                cls._matches_type(
                    key,
                    key_type,
                )
                and cls._matches_type(
                    item,
                    value_type,
                )
                for key, item in value.items()
            )

        # Sequence[T], Mapping[K, V] и прочие generic.
        if isinstance(origin, type):
            return isinstance(
                value,
                origin,
            )

        return True

    @staticmethod
    def _type_name(
        expected_type: Any,
    ) -> str:
        if expected_type is Any:
            return "Any"

        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if origin is None:
            return getattr(
                expected_type,
                "__name__",
                str(expected_type),
            )

        if not args:
            return str(origin)

        return (
            f"{origin}["
            + ", ".join(
                ArgumentDecoder._type_name(arg)
                if arg is not Ellipsis
                else "..."
                for arg in args
            )
            + "]"
        )