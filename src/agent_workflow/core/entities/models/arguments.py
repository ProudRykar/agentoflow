from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

from agent_workflow.core.entities.models.tool import InputT


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


_ARGUMENT_ENVELOPES = frozenset({
    "properties",
    "parameters",
    "arguments",
})


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
        valid: tuple[str, ...] = (),
    ) -> ArgumentDecoderError:
        message = f"Unknown field '{field}'"

        if valid:
            message += (
                ". Valid fields: " + ", ".join(sorted(valid))
            )

        return ArgumentDecoderError(
            message,
            code="unknown_field",
            field=field,
        )


class ArgumentDecoder:
    def decode(
        self,
        data: dict[str, object],
        input_type: type[InputT],
        path: str = "",
    ) -> InputT:
        """Build ``input_type`` from a mapping.

        ``path`` prefixes every field name in an error, so a failure
        inside a nested object says which one.
        """
        if not is_dataclass(input_type):
            return self._decode_opaque(
                data,
                input_type,
            )

        # Some models wrap arguments in a single envelope
        # object (OpenAI-style "properties"). Unwrap exactly
        # one level; deeper nesting stays an error.
        if len(data) == 1:
            sole = next(iter(data))

            if sole in _ARGUMENT_ENVELOPES and isinstance(
                data[sole],
                dict,
            ):
                # Unwrapped only when the name is not a field of this
                # type. "arguments" and "properties" are envelope names
                # *and* ordinary field names, so matching on the name
                # alone meant a tool whose single field was called
                # "arguments" had that field's own value replaced, and
                # every key inside it then read as unknown. If the type
                # declares the name, the caller meant the field.
                if sole not in {
                    field.name
                    for field in fields(input_type)
                }:
                    data = data[sole]

        input_fields = fields(input_type)
        field_names = {
            field.name
            for field in input_fields
        }

        for name in data:
            if name not in field_names:
                raise ArgumentDecoderErrorFactory.unknown_field(
                    f"{path}{name}" if path else name,
                    valid=tuple(sorted(field_names)),
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
                        field=f"{path}{field.name}"
                        if path
                        else field.name,
                    )

                continue

            field_type = hints[field.name]
            value = data[field.name]

            try:
                value = self._decode_value(
                    value,
                    field_type,
                    f"{path}{field.name}."
                    if path
                    else f"{field.name}.",
                )
            except (TypeError, ValueError) as exc:
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field=(
                        f"{path}{field.name}"
                        if path
                        else field.name
                    ),
                    expected=self._type_name(field_type),
                    actual=type(value).__name__,
                ) from exc

            if not self._matches_type(
                value,
                field_type,
            ):
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field=(
                        f"{path}{field.name}"
                        if path
                        else field.name
                    ),
                    expected=self._type_name(field_type),
                    actual=type(value).__name__,
                )

            decoded[field.name] = value

        return input_type(**decoded)

    def _decode_opaque(
        self,
        data: dict[str, object],
        input_type: type[Any],
    ) -> Any:
        """Pass-through for non-dataclass input types.

        Plugins may declare opaque inputs. The raw argument
        mapping is forwarded as ``dict``; primitives are
        coerced from the first value when the tool expects one.
        """

        if input_type is Any or input_type is object:
            return data

        origin = get_origin(input_type)

        if origin in (dict,) or input_type is dict:
            if not isinstance(data, dict):
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field="<input>",
                    expected="dict",
                    actual=type(data).__name__,
                )

            return dict(data)

        if origin is list:
            values = list(data.values())

            return values

        if input_type in (str, bool, int, float):
            values = list(data.values())

            if not values:
                raise ArgumentDecoderErrorFactory.missing_field(
                    field="<input>",
                )

            value = values[0]

            if input_type is str:
                if not isinstance(value, str):
                    raise ArgumentDecoderErrorFactory.invalid_type(
                        field="<input>",
                        expected="str",
                        actual=type(value).__name__,
                    )

                return value

            if input_type is bool:
                if isinstance(value, bool):
                    return value

                raise ArgumentDecoderErrorFactory.invalid_type(
                    field="<input>",
                    expected="bool",
                    actual=type(value).__name__,
                )

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float, str),
            ):
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field="<input>",
                    expected=input_type.__name__,
                    actual=type(value).__name__,
                )

            try:
                return input_type(value)
            except (TypeError, ValueError):
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field="<input>",
                    expected=input_type.__name__,
                    actual=type(value).__name__,
                ) from None

        raise ArgumentDecoderError(
            f"{getattr(input_type, '__name__', input_type)} "
            "must be a dataclass, dict, or a primitive type",
            code="invalid_input_type",
        )

    @staticmethod
    def _at(
        path: str,
        segment: str,
        *,
        separator: str = ".",
    ) -> str:
        """Extend a path with a container segment.

        Paths read as dotted names: ``calls.`` means "inside calls", and
        appending blindly would give ``calls.[1].`` The separator is
        dropped first so an indexed entry reads as one name.

        The joiner differs by segment. An index attaches to the field
        it qualifies, so ``calls[1].``; a mapping key is a separate
        name, so ``others.alpha.``. Joining both the same way is how
        ``calls.[1]`` and ``othersalpha`` come about.
        """

        prefix = path[:-1] if path.endswith(".") else path

        return f"{prefix}{separator}{segment}."

    @classmethod
    def _decode_value(
        cls,
        value: object,
        expected_type: Any,
        path: str = "",
    ) -> object:
        if expected_type is Any:
            return value

        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # A dataclass nested inside another: build it the same way the
        # top level is built, so the two cannot drift apart. Without
        # this the value came back as the raw dict, which looked like
        # a successful decode right up until the handler reached for
        # an attribute that was not there.
        #
        # An instance already of the right type is passed through, so
        # a caller supplying decoded objects is not punished for it.
        if (
            isinstance(expected_type, type)
            and is_dataclass(expected_type)
        ):
            if isinstance(value, expected_type):
                return value

            if not isinstance(value, dict):
                raise ArgumentDecoderErrorFactory.invalid_type(
                    field=path or "<input>",
                    expected=expected_type.__name__,
                    actual=type(value).__name__,
                )

            # ``cls`` is the class here (this is a classmethod) while
            # ``decode`` is an instance method, so the instance has to
            # be made explicitly or the payload lands in ``self``.
            #
            # The path is already threaded through the containers, so
            # a failure below arrives naming the exact item.
            return cls().decode(
                value,
                expected_type,
                path=path,
            )

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

            decoded_items: list[object] = []

            for index, item in enumerate(value):
                decoded_items.append(
                    cls._decode_value(
                        item,
                        args[0],
                        cls._at(path, f"[{index}]", separator=""),
                    )
                )

            return decoded_items

        # set[T]
        if origin is set:
            if not isinstance(value, (set, list, tuple)):
                return value

            if not args:
                return value

            # No index: a set has no order, so "[2]" would name a
            # different member on every run and send the model looking
            # for an entry that is not there.
            return {
                cls._decode_value(
                    item,
                    args[0],
                    path,
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
                        cls._at(path, f"[{index}]", separator=""),
                    )
                    for index, item in enumerate(value)
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
                    cls._at(path, str(key)),
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
            # A list is accepted. JSON has one array type and no
            # tuples, so every tuple-annotated field arrives from a
            # model as a list: rejecting it made *every* tool with a
            # tuple field unusable in practice -- including `remember`
            # with tags, which had been failing the same way.
            if not isinstance(value, (tuple, list)):
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