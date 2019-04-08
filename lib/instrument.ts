// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

export declare namespace Logger {
  export interface SeverityTypeMap {
    verbose: 'v';
    info: 'i';
    warning: 'w';
    error: 'e';
  }

  export type Severity = keyof SeverityTypeMap;

  export type Provider = 'none'|'console';

  /**
   * Logging config that used to control the behavior of logger
   */
  export interface Config {
    /**
     * Specify the logging provider. 'console' by default
     */
    provider?: Provider;
    /**
     * Specify the minimal logger serverity. 'info' by default
     */
    minimalSeverity?: Logger.Severity;
    /**
     * Whether to output date time in log. true by default
     */
    logDateTime?: boolean;
    /**
     * Whether to output source information (Not yet supported). false by default
     */
    logSourceLocation?: boolean;
  }

  export interface CategorizedLogger {
    verbose(content: string): void;
    info(content: string): void;
    warning(content: string): void;
    error(content: string): void;
  }
}

export interface Logger {
  (category: string): Logger.CategorizedLogger;

  verbose(content: string): void;
  verbose(category: string, content: string): void;
  info(content: string): void;
  info(category: string, content: string): void;
  warning(content: string): void;
  warning(category: string, content: string): void;
  error(content: string): void;
  error(category: string, content: string): void;

  /**
   * Reset the logger configuration.
   * @param config specify an optional default config
   */
  reset(config?: Logger.Config): void;
  /**
   * Set the logger's behavior on the given category
   * @param category specify a category string. If '*' is specified, all previous configuration will be overwritten. If
   * '' is specified, the default behavior will be updated.
   * @param config the config object to indicate the logger's behavior
   */
  set(category: string, config: Logger.Config): void;
}

interface LoggerProvider {
  log(severity: Logger.Severity, content: string, category?: string): void;
}
class NoOpLoggerProvider implements LoggerProvider {
  log(severity: Logger.Severity, content: string, category?: string) {
    // do nothing
  }
}
class ConsoleLoggerProvider implements LoggerProvider {
  log(severity: Logger.Severity, content: string, category?: string) {
    console.log(`${this.color(severity)} ${category ? '\x1b[35m' + category + '\x1b[0m ' : ''}${content}`);
  }

  private color(severity: Logger.Severity) {
    switch (severity) {
      case 'verbose':
        return '\x1b[34;40mv\x1b[0m';
      case 'info':
        return '\x1b[32mi\x1b[0m';
      case 'warning':
        return '\x1b[30;43mw\x1b[0m';
      case 'error':
        return '\x1b[31;40me\x1b[0m';
      default:
        throw new Error(`unsupported severity: ${severity}`);
    }
  }
}

const SEVERITY_VALUE = {
  verbose: 1000,
  info: 2000,
  warning: 4000,
  error: 5000
};

const LOGGER_PROVIDER_MAP: {readonly [provider: string]: Readonly<LoggerProvider>} = {
  ['none']: new NoOpLoggerProvider(),
  ['console']: new ConsoleLoggerProvider()
};
const LOGGER_DEFAULT_CONFIG = {
  provider: 'console',
  minimalSeverity: 'info',
  logDateTime: true,
  logSourceLocation: false
};
let LOGGER_CONFIG_MAP:
    {[category: string]: Readonly<Required<Logger.Config>>} = {['']: LOGGER_DEFAULT_CONFIG as Required<Logger.Config>};

function log(category: string): Logger.CategorizedLogger;
function log(severity: Logger.Severity, content: string): void;
function log(severity: Logger.Severity, category: string, content: string): void;
function log(severity: Logger.Severity, arg1: string, arg2?: string): void;
function log(
    arg0: string|Logger.Severity, arg1?: string, arg2?: string|number, arg3?: number): Logger.CategorizedLogger|void {
  if (arg1 === undefined) {
    // log(category: string): Logger.CategorizedLogger;
    return createCategorizedLogger(arg0);
  } else if (arg2 === undefined) {
    // log(severity, content);
    logInternal(arg0 as Logger.Severity, arg1, 1);
  } else if (typeof arg2 === 'number' && arg3 === undefined) {
    // log(severity, content, stack)
    logInternal(arg0 as Logger.Severity, arg1, arg2);
  } else if (typeof arg2 === 'string' && arg3 === undefined) {
    // log(severity, category, content)
    logInternal(arg0 as Logger.Severity, arg2, 1, arg1);
  } else if (typeof arg2 === 'string' && typeof arg3 === 'number') {
    // log(severity, category, content, stack)
    logInternal(arg0 as Logger.Severity, arg2, arg3, arg1);
  } else {
    throw new TypeError('input is valid');
  }
}

function createCategorizedLogger(category: string): Logger.CategorizedLogger {
  return {
    verbose: log.verbose.bind(null, category),
    info: log.info.bind(null, category),
    warning: log.warning.bind(null, category),
    error: log.error.bind(null, category)
  };
}

// NOTE: argument 'category' is put the last parameter beacause typescript
// doesn't allow optional argument put in front of required argument. This
// order is different from a usual logging API.
function logInternal(severity: Logger.Severity, content: string, stack: number, category?: string) {
  const config = LOGGER_CONFIG_MAP[category || ''] || LOGGER_CONFIG_MAP[''];
  if (SEVERITY_VALUE[severity] < SEVERITY_VALUE[config.minimalSeverity]) {
    return;
  }

  if (config.logDateTime) {
    content = `${new Date().toISOString()}|${content}`;
  }

  if (config.logSourceLocation) {
    // TODO: calculate source location from 'stack'
  }

  LOGGER_PROVIDER_MAP[config.provider].log(severity, content, category);
}

// tslint:disable-next-line:no-namespace
namespace log {
  export function verbose(content: string): void;
  export function verbose(category: string, content: string): void;
  export function verbose(arg0: string, arg1?: string) {
    log('verbose', arg0, arg1);
  }
  export function info(content: string): void;
  export function info(category: string, content: string): void;
  export function info(arg0: string, arg1?: string) {
    log('info', arg0, arg1);
  }
  export function warning(content: string): void;
  export function warning(category: string, content: string): void;
  export function warning(arg0: string, arg1?: string) {
    log('warning', arg0, arg1);
  }
  export function error(content: string): void;
  export function error(category: string, content: string): void;
  export function error(arg0: string, arg1?: string) {
    log('error', arg0, arg1);
  }

  export function reset(config?: Logger.Config): void {
    LOGGER_CONFIG_MAP = {};
    // tslint:disable-next-line:no-backbone-get-set-outside-model
    set('', config || {});
  }
  export function set(category: string, config: Logger.Config): void {
    if (category === '*') {
      reset(config);
    } else {
      const previousConfig = LOGGER_CONFIG_MAP[category] || LOGGER_DEFAULT_CONFIG;
      LOGGER_CONFIG_MAP[category] = {
        provider: config.provider || previousConfig.provider,
        minimalSeverity: config.minimalSeverity || previousConfig.minimalSeverity,
        logDateTime: (config.logDateTime === undefined) ? previousConfig.logDateTime : config.logDateTime,
        logSourceLocation: (config.logSourceLocation === undefined) ? previousConfig.logSourceLocation :
                                                                      config.logSourceLocation
      };
    }

    // TODO: we want to support wildcard or regex?
  }
}

// tslint:disable-next-line:variable-name
export const Logger: Logger = log;

export declare namespace Profiler {
  export interface Config {
    maxNumberEvents?: number;
    flushBatchSize?: number;
    flushIntervalInMilliseconds?: number;
  }

  export type EventCategory = 'session'|'node'|'op'|'backend';

  export interface Event {
    end(): void;
  }
}

class Event implements Profiler.Event {
  constructor(
      public category: Profiler.EventCategory, public name: string, public startTime: number,
      private endCallback: (e: Event) => void) {}

  end() {
    this.endCallback(this);
  }
}

class EventRecord {
  constructor(
      public category: Profiler.EventCategory, public name: string, public startTime: number, public endTime: number) {}
}

export class Profiler {
  static create(config?: Profiler.Config): Profiler {
    if (config === undefined) {
      return new this();
    }
    return new this(config.maxNumberEvents, config.flushBatchSize, config.flushIntervalInMilliseconds);
  }

  private constructor(maxNumberEvents?: number, flushBatchSize?: number, flushIntervalInMilliseconds?: number) {
    this._started = false;
    this._maxNumberEvents = maxNumberEvents === undefined ? 10000 : maxNumberEvents;
    this._flushBatchSize = flushBatchSize === undefined ? 10 : flushBatchSize;
    this._flushIntervalInMilliseconds = flushIntervalInMilliseconds === undefined ? 5000 : flushIntervalInMilliseconds;
  }

  // start profiling
  start() {
    this._started = true;
    this._timingEvents = [];
    this._flushTime = now();
    this._flushPointer = 0;
  }

  // stop profiling
  stop() {
    this._started = false;

    const nodeTimings: {
      name: string,
      time: number,
      setInputTime: number,
      computeTime: number,
      count?: number,
    }[] = [];

    for (let i = this._timingEvents.length - 1; i >= 0; i--) {
      const event = this._timingEvents[i];
      const elapsedTime = event.endTime - event.startTime;
      switch (event.category) {
        case 'op':
        case 'session':
          continue;
        case 'node': {
          nodeTimings.unshift({
            name: event.name,
            time: elapsedTime,
            setInputTime: 0,
            computeTime: 0
          });
        } break;
        case 'backend': {
          switch (event.name) {
            case 'WebNN.Execution.setInput': {
              nodeTimings[0].setInputTime += elapsedTime;
            } break;
            case 'WebNN.Execution.startCompute': {
              nodeTimings[0].computeTime += elapsedTime;
            } break;
          }
        } break;
      }
    }

    const reducedNodeTimings = [];
    const mapping: Map<string, number> = new Map();
    for (const i in nodeTimings) {
      const nodeTiming = nodeTimings[i];
      if (mapping.has(nodeTiming.name)) {
        const index = mapping.get(nodeTiming.name)!;
        reducedNodeTimings[index].time += nodeTiming.time;
        reducedNodeTimings[index].setInputTime += nodeTiming.setInputTime;
        reducedNodeTimings[index].computeTime += nodeTiming.computeTime;
        reducedNodeTimings[index].count!++;
      } else {
        nodeTiming.count = 1;
        reducedNodeTimings.push(nodeTiming);
        mapping.set(nodeTiming.name, parseInt(i));
      }
    }

    for (const node of reducedNodeTimings) {
      const time = node.time / node.count!;
      const setInputTime = node.setInputTime / node.count!;
      const computeTime = node.computeTime / node.count!;
      const otherTime = time - setInputTime - computeTime;
      const padNum = node.name.length <= 30 ? 30 - node.name.length : 0;
      let str = `${node.name}:${' '.repeat(padNum)} total ${time.toFixed(5).slice(0, 6)}`;
      if (setInputTime !== 0 || computeTime !== 0) {
        str += `, \
setInput ${setInputTime.toFixed(5).slice(0, 6)}, \
computeTime ${computeTime.toFixed(5).slice(0, 6)}, \
reorder+overhead ${otherTime.toFixed(5).slice(0, 6)}`;
      }
      console.log(str);
    }
  }

  // create an event scope for the specific function
  event<T>(category: Profiler.EventCategory, name: string, func: () => T): T;
  event<T>(category: Profiler.EventCategory, name: string, func: () => Promise<T>): Promise<T>;

  event<T>(category: Profiler.EventCategory, name: string, func: () => T | Promise<T>): T|Promise<T> {
    const event = this._started ? this.begin(category, name) : undefined;
    let isPromise = false;

    try {
      const res = func();

      // we consider a then-able object is a promise
      if (res && typeof (res as Promise<T>).then === 'function') {
        isPromise = true;
        return new Promise<T>((resolve, reject) => {
          (res as Promise<T>)
              .then(
                  value => {  // fulfilled
                    resolve(value);
                    if (event) {
                      event.end();
                    }
                  },
                  reason => {  // rejected
                    reject(reason);
                    if (event) {
                      event.end();
                    }
                  });
        });
      }

      return res;

    } finally {
      if (!isPromise && event) {
        event.end();
      }
    }
  }

  // begin an event
  begin(category: Profiler.EventCategory, name: string): Event {
    if (!this._started) {
      throw new Error('profiler is not started yet');
    }
    const startTime = now();
    this.flush(startTime);
    return new Event(category, name, startTime, e => this.end(e));
  }

  // end the specific event
  private end(event: Event) {
    if (this._timingEvents.length < this._maxNumberEvents) {
      const endTime = now();
      this._timingEvents.push(new EventRecord(event.category, event.name, event.startTime, endTime));
      this.flush(endTime);
    }
  }

  private logOneEvent(event: EventRecord) {
    Logger.verbose(
        `Profiler.${event.category}`,
        `${(event.endTime - event.startTime).toFixed(2)}ms on event '${event.name}' at ${event.endTime.toFixed(2)}`);
  }

  private flush(currentTime: number) {
    if (this._timingEvents.length - this._flushPointer >= this._flushBatchSize ||
        currentTime - this._flushTime >= this._flushIntervalInMilliseconds) {
      // should flush when either batch size accumlated or interval elepsed

      for (const previousPointer = this._flushPointer; this._flushPointer < previousPointer + this._flushBatchSize &&
           this._flushPointer < this._timingEvents.length;
           this._flushPointer++) {
        this.logOneEvent(this._timingEvents[this._flushPointer]);
      }

      this._flushTime = now();
    }
  }

  get started() {
    return this._started;
  }
  private _started = false;
  private _timingEvents: EventRecord[];

  private readonly _maxNumberEvents: number;

  private readonly _flushBatchSize: number;
  private readonly _flushIntervalInMilliseconds: number;

  private _flushTime: number;
  private _flushPointer = 0;
}

/**
 * returns a number to represent the current timestamp in a resolution as high as possible.
 */
export const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : Date.now;
